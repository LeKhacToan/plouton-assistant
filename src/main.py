import json
import re
import os
import mysql.connector
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")

def connect_db():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )

def init_chroma_client():
    chroma_host = os.getenv("CHROMA_HOST", "chroma")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    return chromadb.HttpClient(
        host=chroma_host,
        port=chroma_port,
    )

chroma_client = init_chroma_client()
chroma_client.heartbeat()
collection = chroma_client.create_collection(name="books", get_or_create=True)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def call_openai(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def sync_books_to_chroma():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT b.id, b.title, b.shelf_id, s.name
        FROM book b 
        JOIN shelf s ON b.shelf_id = s.id
    """)
    books = cursor.fetchall()
    
    for book_id, title, shelf_id, shelf_name in books:
        embedding = embedder.encode(title).tolist()
        collection.upsert(
            ids=[str(book_id)],
            embeddings=[embedding],
            metadatas=[{
                "book_id": book_id,
                "title": title,
                "shelf_id": shelf_id,
                "shelf_name": shelf_name
            }]
        )
    cursor.close()
    conn.close()

def find_book_in_chroma(query):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    if results['ids'][0]:
        return results['metadatas'][0][0]
    return None

def move_book(book_id, new_shelf_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM shelf WHERE id = %s", (new_shelf_id,))
    if not cursor.fetchone():
        cursor.close()
        conn.close()
        return f"Kệ {new_shelf_id} không tồn tại."
    
    update_query = "UPDATE book SET shelf_id = %s WHERE id = %s"
    cursor.execute(update_query, (new_shelf_id, book_id))
    conn.commit()
    affected_rows = cursor.rowcount
    cursor.close()
    conn.close()
    sync_books_to_chroma()
    return f"Đã di chuyển sách sang kệ {new_shelf_id}." if affected_rows > 0 else "Không tìm thấy sách."

def query_books_by_prefix(prefix):
    conn = connect_db()
    cursor = conn.cursor()
    # Case-insensitive search
    cursor.execute("""
        SELECT b.title, s.name
        FROM book b
        JOIN shelf s ON b.shelf_id = s.id
        WHERE LOWER(b.title) LIKE LOWER(%s)
    """, (f"{prefix}%",))
    books = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not books:
        return "Không tìm thấy sách nào với tiêu đề bắt đầu bằng '{}'.".format(prefix)
    
    result = "Các sách với tiêu đề bắt đầu bằng '{}':\n".format(prefix)
    for title, shelf_name in books:
        result += f"- '{title}' trên kệ '{shelf_name}'\n"
    return result.strip()

def process_query(query):
    book_info = find_book_in_chroma(query)
    if not book_info:
        return "Không tìm thấy sách phù hợp với câu hỏi."
    
    book_id = book_info["book_id"]
    title = book_info["title"]
    shelf_id = book_info["shelf_id"]
    shelf_name = book_info["shelf_name"]
    
    context = f"""
    Thông tin sách:
    - ID: {book_id}
    - Tiêu đề: {title}
    - Kệ ID: {shelf_id}
    - Tên kệ: {shelf_name}
    
    Câu hỏi người dùng: "{query}"
    """
    
    prompt = f"""
    Dựa trên ngữ cảnh sau, trả lời câu hỏi một cách tự nhiên hoặc thực hiện hành động được yêu cầu:
    {context}
    
    Nếu câu hỏi yêu cầu tra cứu vị trí sách, trả về câu trả lời dạng: "Sách [title] đang ở kệ [shelf_name] (ID: [shelf_id])."
    Nếu câu hỏi yêu cầu di chuyển sách, trích xuất ID kệ mới và trả về JSON:
    ```json
    {{"intent": "move_book", "book_id": {book_id}, "new_shelf_id": "ID_KỆ_MỚI"}}
    ```
    Nếu câu hỏi hỏi về thông tin sách (ví dụ: có sách nào bắt đầu bằng một từ cụ thể), trích xuất tiền tố tiêu đề và trả về JSON:
    ```json
    {{"intent": "query_books", "prefix": "TIỀN_TỐ"}}
    ```
    Nếu câu hỏi không liên quan đến các vấn đề trên hãy trả lời một cách tự nhiên
    Nếu không hiểu câu hỏi, trả về: "Không hiểu câu hỏi."
    """
    
    try:
        llm_output = call_openai(prompt)
        json_match = re.search(r'```json\n(.+?)\n```', llm_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            if result.get("intent") == "move_book":
                new_shelf_id = result.get("new_shelf_id")
                return move_book(book_id, new_shelf_id)
            elif result.get("intent") == "query_books":
                prefix = result.get("prefix")
                if not prefix or not isinstance(prefix, str):
                    return "Tiền tố tiêu đề không hợp lệ."
                return query_books_by_prefix(prefix)
        else:
            return llm_output.strip()
    except Exception as e:
        return f"Lỗi xử lý câu hỏi: {str(e)}"

def main():
    sync_books_to_chroma()
    while True:
        user_input = input("Nhập câu hỏi (hoặc 'thoát' để dừng): ")
        encoded_text = user_input.encode('utf-8', 'ignore').decode('utf-8')
        print(encoded_text)
        if user_input.lower() == "thoát":
            break
        response = process_query(encoded_text)
        print(response)

if __name__ == "__main__":
    main()
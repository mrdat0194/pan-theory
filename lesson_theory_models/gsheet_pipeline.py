import sys
import os
import polars as pl
from functools import reduce

# Add Gsheet_util directory to sys.path so we can import it
sys.path.append(r"C:\Users\mrdat\PycharmProjects\pan-theory\main_def\ggl_api")
import Gsheet_util

# --- 1. CLOSURE THEORY (Tương tự def outter / inner) ---
def make_gsheet_reader(spreadsheet_id):
    """
    Closure: Lưu trữ spreadsheet_id ở scope bên ngoài (outter).
    Trả về hàm read_sheet (inner) nhận vào sheet identifier.
    """
    def read_sheet(sheet_id_info):
        # Hàm inner nhận dict chứa GID và Name của sheet
        gid = sheet_id_info["gid"]
        name = sheet_id_info["name"]
        print(f"-> Reading sheet: {name} (GID: {gid}) from {spreadsheet_id}...")
        
        try:
            # Lưu ý: File ID 1-Kip2_MicfV0ebRZX8N5vhj3b-8RxuQp hiện là định dạng .xlsx 
            # gspread API sẽ báo lỗi 400. Cần Save as "Google Sheets" trên web để chạy thành công.
            df = Gsheet_util.spreadsheet_to_df(id=spreadsheet_id, name_worksheet=name)
            return df
        except Exception as e:
            print(f"Error reading {name}: {e}")
            return pl.DataFrame()
            
    return read_sheet


# --- 2. HÀM RIÊNG BIỆT (Tương tự def is_even, square) ---
def is_not_empty(df: pl.DataFrame) -> bool:
    """Hàm dùng cho filter() để loại bỏ các Dataframe rỗng"""
    return df is not None and not df.is_empty()


if __name__ == "__main__":
    # --- 3. DỮ LIỆU BAN ĐẦU ---
    SPREADSHEET_ID = "1-Kip2_MicfV0ebRZX8N5vhj3b-8RxuQp"
    
    # URL 1: ...#gid=1380481769
    # URL 2: ...#gid=48503281
    # URL 3: ...#gid=1600003352
    # Vì gspread API hoạt động theo tên sheet thay vì GID, ta map GID vào tên sheet tương ứng
    # (Thay thế Name bằng tên thật trong file Excel của bạn)
    target_sheets = [
        {"gid": "1380481769", "name": "Sheet1"},
        {"gid": "48503281",   "name": "Sheet2"},
        {"gid": "1600003352", "name": "Sheet3"}
    ]

    # Khởi tạo reader closure
    sheet_reader = make_gsheet_reader(SPREADSHEET_ID)

    # --- 4. THỰC HIỆN CHUỖI XỬ LÝ DỮ LIỆU (Functional Programming: map, filter, reduce) ---
    print("\n--- Starting Functional Data Reading Pipeline ---")
    
    # map(sheet_reader, target_sheets): Áp dụng hàm đọc lên từng sheet
    # filter(is_not_empty, ...): Giữ lại những sheet đọc thành công và có dữ liệu
    valid_dataframes = list(filter(is_not_empty, map(sheet_reader, target_sheets)))

    print(f"\nResult: Successfully read {len(valid_dataframes)} valid sheets.")
    
    if valid_dataframes:
        # Nếu muốn gộp tất cả thành 1 bảng duy nhất (giống hàm sum() trong ví dụ lessonLearnt)
        combined_df = reduce(lambda df1, df2: pl.concat([df1, df2], how="vertical"), valid_dataframes)
        print("Size after concatenation:", combined_df.shape)
    else:
        print("No valid DataFrames were returned. Please ensure the Google Drive file is converted to a native Google Sheet format.")

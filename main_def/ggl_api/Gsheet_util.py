import gspread
import time
import random
import polars as pl
from typing import Any


import os
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'kasa', 'ga4-gtm-automation.json')
gc = gspread.service_account(json_path)
""" from my_function.Gsheet_util import *"""
def share_gsheet_anyone_with_link_viewer(spreadsheet_id: str):
    """
    Share a Google Sheet with anyone who has the link as a viewer.

    Args:
        spreadsheet_id (str): The ID of the Google Spreadsheet.
    """
    sh = gc.open_by_key(spreadsheet_id)
    sh.share(None, perm_type='anyone', role='reader', notify=False)

# share_gsheet_anyone_with_link_viewer('1WtxhvOkQwqa7Tevz3VsbB4Pj2T9hbgN_v68a_QfNVBo')


def safe_gsheet_request(some_operation:Any):
    def f( *arg, **kwarg):
        # research to make it arg
        max_retries=15
        base_delay=1
        max_delay=32
        retries = 0
        t = 0
        while retries < max_retries:
            try:
                # Simulate an operation that might fail, like an API call or network request
                # if callable(some_operation):
                result = some_operation(*arg, **kwarg)
                # else: result = some_operation  # Success
                return result  # If operation is successful, return the result
            except gspread.exceptions.APIError as e:  # Replace with more specific exceptions as needed
                retries += 1
                t+= 1
                random_number_milliseconds = random.randint(0, 1000) / 1000
                wait_time = min( t**(base_delay + random_number_milliseconds) , max_delay)
                print(f"Attempt {retries} failed, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        raise gspread.exceptions.APIError(f"Failed after {max_retries} retries.")
    return f

#Sub function
def insert_at_idx(df:pl.DataFrame, index:int, *args, **kwargs) -> None:
    # This function is to add new column and assign index for Polars Dataframe
    # Example: df = df.insert_at_idx(0, Client=pl.lit(name))
    """pl.DataFrame.with_at_idx=with_at_idx"""
    if len(args)+len(kwargs)>1:
        raise ValueError("Only one new column allowed")
    # loops will continue to work but then you need to deal 
    # with precedent between args and kwargs
    cols=df.columns
    for arg in args:
        cols.insert(index, arg)
    for colname, arg in kwargs.items():
        cols.insert(index, arg.alias(colname))
    return df.select(cols)

pl.DataFrame.insert_at_idx = insert_at_idx

#Sub function 
def to_numeric(s: pl.Series) -> pl.Series:
    # This function is to categorize datatype in dataframe
    # Example: df = pl.select(to_numeric(s)for s in df)
    try:
        result = s.cast(pl.Int64)
    except pl.exceptions.InvalidOperationError:
        try:result = s.cast(pl.Float64)
        except pl.exceptions.InvalidOperationError:
            result = s.cast(pl.String)
    return result

#Sub function
def to_float(s: pl.Series) -> pl.Series:
    # This function is to categorize datatype in dataframe
    # Example: df = pl.select(to_numeric(s)for s in df)
    try:result = s.cast(pl.Float64)
    except pl.exceptions.InvalidOperationError:
        result = s.cast(pl.String)
    return result



#Sub function 
@safe_gsheet_request
def spreadsheet_to_df(id:str, idx_range:str="" ,idx_cell: str = "A1", idx_worksheet: int = 0, name_worksheet: str = None) -> pl.DataFrame:
    
    # Example: spreadsheet_to_df(id="14TIFFnlhHSJIkZocZhNOZp7UqtqKdPSoYaZPCmhCD_k",  idx_cell:"A2",idx_worksheet: 1, name_worksheet: "Sheet1")
    """ 
    id       : id in url of website
    idx_cell : start Cell
    idx_worksheet: order of workbook in spreadsheet
    name_worksheet: name of the worksheet
    """
    
    if name_worksheet is not None:
        worksheet_list = [sheet.title for sheet in gc.open_by_key(id).worksheets()]
        # return enumerate(worksheet_list)
        try:
            
            idx_worksheet = next((i for i, title in enumerate(worksheet_list) if title == name_worksheet), None)
            
            idx_worksheet = str(idx_worksheet)
            
            
            if idx_worksheet is None:
                # return 'aaaa'
                raise ValueError(f"Worksheet {name_worksheet} not found")
        except ValueError:
            # return 'bbbb'
            print(f"Worksheet {name_worksheet} not found")
            return pl.DataFrame()

    if idx_range != "":
        try:
            sh_name = gc.open_by_key(id).get_worksheet(index= int(idx_worksheet))
            try:
                df      = pl.DataFrame(sh_name.get(idx_range,pad_values=True)).transpose()
                df      = df.rename(dict(zip(df.columns, df.head(1).transpose().to_series()))).slice(1)        
                # Before calculate optimize datatype 
                df      = pl.select(to_numeric(s)for s in df)
            except pl.exceptions.NoDataError:
                df = pl.DataFrame()    
        except TypeError :
            df = pl.DataFrame()
    else:        
        try:
            # return 'sh_name'
            sh_name     = gc.open_by_key(id).get_worksheet(index= int(idx_worksheet))
            
            try:
                df      = pl.DataFrame(sh_name.get(f"{idx_cell}:XFD",pad_values=True)).transpose()
                df      = df.rename(dict(zip(df.columns, df.head(1).transpose().to_series()))).slice(1) 
                      
                # Before calculate optimize datatype 
                df      = pl.select(to_numeric(s)for s in df)
                
            except pl.exceptions.NoDataError:
                df = pl.DataFrame()    
        except TypeError:
            df = pl.DataFrame()
            
    return df

#Sub function 
@safe_gsheet_request
def clear_worksheet(id: str,   idx_cell:str = "A1" , name_worksheet: str = None, idx_worksheet:  int = 0) -> Any:
    # Example: clear_worksheet(id="14TIFFnlhHSJIkZocZhNOZp7UqtqKdPSoYaZPCmhCD_k", name_worksheet= "MASTER BRAND SPENDING", idx_cell= "A2", idx_worksheet= 1)
    """ 
    Rule: worksheet will clear by defined row 
    """
    if name_worksheet != None:
        worksheet_list= [sheet.title for sheet in gc.open_by_key(id).worksheets() if name_worksheet != None]
        try:
            idx_worksheet = worksheet_list.index(name_worksheet) 
            print(f"Clear worksheet {name_worksheet} at index {idx_worksheet}")
        except ValueError:
            print(f"Worksheet {name_worksheet} not found")
            pass
    target_sh = gc.open_by_key(id).get_worksheet(index= int(idx_worksheet))
    target_sh.batch_clear([f"{idx_cell}:XFD"])

    return None

@safe_gsheet_request
def del_worksheet(id: str,   idx_cell:str = "A1" , name_worksheet: str = None, idx_worksheet:  int = 0) -> Any:
    # Example: clear_worksheet(id="14TIFFnlhHSJIkZocZhNOZp7UqtqKdPSoYaZPCmhCD_k", name_worksheet= "MASTER BRAND SPENDING", idx_cell= "A2", idx_worksheet= 1)
    """ 
    Rule: worksheet will clear by defined row 
    """
    if name_worksheet != None:
        worksheet_list= [sheet.title for sheet in gc.open_by_key(id).worksheets() if name_worksheet != None]
        try:
            # print(worksheet_list)
            idx_worksheet = worksheet_list.index(name_worksheet) 
            print(f"Clear worksheet {name_worksheet} at index {idx_worksheet}")
        except ValueError:
            print(f"Worksheet {name_worksheet} not found")
            pass
    # target_sh = 
    sh_name = gc.open_by_key(id).get_worksheet(index= int(idx_worksheet))
    gc.open_by_key(id).del_worksheet(sh_name)
    # target_sh.del_worksheet(target_sh)
    return None

#Sub function
@safe_gsheet_request
def df_append_spreadsheet(id: str, df: pl.DataFrame, name_worksheet: str = None, idx_worksheet: int = None, header: bool = False) -> Any:
    # Example: df_to_spreadsheet(id="14TIFFnlhHSJIkZocZhNOZp7UqtqKdPSoYaZPCmhCD_k", name_worksheet: "Test Append Sheet", df = df, idx_worksheet = 1 )
    # Note:    This function use client memory to apppend data 
    """
    Rule: dataframe will auto find last row to append data in spreadsheet
    """
    if df.is_empty() | any(df.row(0)) == False:
        print("No data to append")
        pass
    if idx_worksheet != None or name_worksheet == None:
        name_worksheet= [sheet.title for sheet in gc.open_by_key(id).worksheets()][idx_worksheet]
    target_sh = gc.open_by_key(id)
    # Announce
    print(f"Append data to {name_worksheet}")
    if header == True:
        target_sh.values_append(name_worksheet, {'valueInputOption': 'USER_ENTERED'}, {'values': [((df.columns))]})
    target_sh.values_append(name_worksheet, {'valueInputOption': 'USER_ENTERED'}, {'values': df.rows()})
    return None

#Sub function
@safe_gsheet_request
def df_write_spreadsheet(id: str, df: pl.DataFrame, name_worksheet: str = None, idx_worksheet: int = 0, idx_cell: str = "A1", header: bool = False) -> Any:
    # Example: df_update_spreadsheet(id="14TIFFnlhHSJIkZocZhNOZp7UqtqKdPSoYaZPCmhCD_k", name_worksheet= "Test Append Sheet", df = df,idx_cell = "B3",name_worksheet: "aaa" )
    """ 
    Rule: spreadsheet will be replaced by defined dataframe 
    """
    try:
        if name_worksheet != None:
            target_sh = gc.open_by_key(id)
            worksheet_list = [sheet.title for sheet in target_sh.worksheets() if name_worksheet != None]
            idx_worksheet  = worksheet_list.index(name_worksheet)
            
            worksheet      = target_sh.get_worksheet(index= int(idx_worksheet))
        else:
            target_sh = gc.open_by_key(id)
            worksheet = target_sh.get_worksheet(index=int(idx_worksheet))
    except:

        worksheet = target_sh.add_worksheet(title= name_worksheet, rows= df.shape[0], cols=df.shape[1])    
       
    # Clear worksheet
    if header == True:
        clear_worksheet(id=id, name_worksheet=name_worksheet, idx_cell=idx_cell, idx_worksheet=idx_worksheet)
        # Append the column titles (header) of the polars DataFrame
        print(list(df.columns))
        worksheet.update([list(df.columns)], idx_cell)
        # Logic to increment the row of a cell reference (e.g., "A1" becomes "A2")
        col_letters = ""
        row_number_str = ""
        for char in idx_cell:
            if char.isalpha():
                col_letters += char
            elif char.isdigit():
                row_number_str += char
            # Assuming idx_cell is always in a valid "A1" format, no other chars expected
        
        next_row_cell = f"{col_letters}{int(row_number_str) + 1}"
        worksheet.update(list(df.rows()), f"{next_row_cell}:XFD")    
    else:
        clear_worksheet(id=id, name_worksheet=name_worksheet, idx_cell=idx_cell, idx_worksheet=idx_worksheet)
        worksheet.update(list(df.rows()), f"{idx_cell}:XFD")

def df_append_to_spreadsheet(id: str, df: pl.DataFrame, name_worksheet: str = None, idx_worksheet: int = 0) -> Any:
    try:
        if name_worksheet != None:
            target_sh = gc.open_by_key(id)
            worksheet_list = [sheet.title for sheet in target_sh.worksheets() if name_worksheet != None]
            idx_worksheet  = worksheet_list.index(name_worksheet)
            worksheet      = target_sh.get_worksheet(index= int(idx_worksheet))
        else:
            target_sh = gc.open_by_key(id)
            worksheet = target_sh.get_worksheet(index=int(idx_worksheet))
    except:
        worksheet = target_sh.add_worksheet(title= name_worksheet, rows= df.shape[0], cols=df.shape[1])  

    
    return target_sh.values_append(worksheet.title, {'valueInputOption': 'USER_ENTERED'}, {'values': df.rows()})


@safe_gsheet_request
def copy_sheet_tab(from_sheet,to_sheet, sheet_id, newName):

    # The ID of the sheet to copy. Everybody has access!!!
    # Get source spreadsheet
    source_spreadsheet = gc.open_by_key(from_sheet)
    
    # Get destination spreadsheet
    dest_spreadsheet = gc.open_by_key(to_sheet)

    # Copy the sheet to destination spreadsheet
    worksheet = source_spreadsheet.get_worksheet_by_id(sheet_id)
    copied_sheet = source_spreadsheet._spreadsheets_sheets_copy_to(
        sheet_id,
        dest_spreadsheet.id
    )

    # Rename the copied sheet
    body = {
        'requests': [{
            'updateSheetProperties': {
                'properties': {
                    'sheetId': copied_sheet['sheetId'],
                    'title': newName
                },
                'fields': 'title'
            }
        }]
    }
    dest_spreadsheet.batch_update(body)

    print("Done " + str(sheet_id))

#Sub function
@safe_gsheet_request
def df_update_cells_batch(id: str, name_worksheet: str, df: Any, row_index: int, updates: dict) -> Any:
    # Updates specific cells in a row (identified by row_index in dataframe) based on column names using batch_update.
    # Preserves other cells in the row.
    # 
    # Args:
    #     id (str): Spreadsheet ID
    #     name_worksheet (str): Name of the worksheet
    #     df (Any): The current DataFrame (Polars or Pandas) corresponding to the sheet
    #     row_index (int): 0-based index of the row in the DataFrame. 
    #                      (Actual Sheet Row will be row_index + 2 assuming 1 header row)
    #     updates (dict): Dictionary mapping column names to new values {col_name: new_value}
    
    target_sh = gc.open_by_key(id)
    worksheet = target_sh.worksheet(name_worksheet)
    
    batch_data = []
    
    # Helper to get column letter
    def col_to_letter(idx):
        dividend = idx + 1
        column_label = ""
        while dividend > 0:
            modulo = (dividend - 1) % 26
            column_label = chr(65 + modulo) + column_label
            dividend = (dividend - modulo - 1) // 26
        return column_label

    cols = list(df.columns)
    for col_name, value in updates.items():
        if col_name in cols:
            col_idx = cols.index(col_name)
            col_letter = col_to_letter(col_idx)
            # Row index in sheet is row_index + 2 (1-for-header + 1-for-0-based)
            cell_address = f'{col_letter}{row_index + 2}'
            
            batch_data.append({
                'range': cell_address,
                'values': [[str(value)]]
            })
        else:
            print(f"Warning: Column '{col_name}' not found in dataframe - skipping update for this field.")

    if batch_data:
        worksheet.batch_update(batch_data)
        print(f"Successfully updated {len(batch_data)} cells via batch_update.")
    else:
        print('No values to update.')
    
    return None

import os
import gzip
from PIL import Image
import fitz  # PyMuPDF
import svgwrite
import cairosvg
import re
from bs4 import BeautifulSoup
import pandas as pd
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

def extract_vectors_from_pdf(pdf_file_path, output_dir):
    subdir_for_svg = 'svg/'
    subdir_for_png = 'png/'
    # Create subdirectories if they don't exist
    svg_dir = os.path.join(output_dir, subdir_for_svg)
    png_dir = os.path.join(output_dir, subdir_for_png)
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    fname = os.path.splitext(os.path.basename(pdf_file_path))[0]
    doc = fitz.open(pdf_file_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        width, height = page.rect.width, page.rect.height
        vectors = page.get_drawings(extended=True)

        #PAths to save
        svg_path = os.path.join(svg_dir, f"{fname}_p{page_number}.svg")
        png_path = os.path.join(png_dir, f"{fname}_p{page_number}.png")
        cropped_png_path = os.path.join(png_dir, f"{fname}_p{page_number}_crop.png")
        compressed_svg_path = os.path.join(svg_dir, f"{fname}_p{page_number}.svg.gz")

        dwg = svgwrite.Drawing(svg_path, size=(width, 1.35 * height))
        for vector in vectors:
            path_items = vector['items']
            for cmd, *points in path_items:
                if vector['color'] == (0.0, 0.0, 0.0):
                    dwg.add(dwg.line((points[0][0], points[0][1]), (points[1][0], points[1][1]), stroke=svgwrite.rgb(0, 0, 0, '%')))
        dwg.save()
        #Convert to png
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        crop_whitespace(png_path, cropped_png_path)
        #Compress svg
        with open(svg_path, 'rb') as f_in:
            with gzip.open(compressed_svg_path, 'wb') as f_out:
                f_out.writelines(f_in)
        # Remove original SVG (too big!)
        os.remove(svg_path)

def crop_whitespace(input_image_path, output_image_path, threshold=10):
    image = Image.open(input_image_path)
    image_data = image.getdata()
    image_size = image.size

    # Find bounding box of non-white pixels
    min_x = image_size[0]
    min_y = image_size[1]
    max_x = 0
    max_y = 0

    for y in range(image_size[1]):
        for x in range(image_size[0]):
            pixel = image_data[y * image_size[0] + x]
            if pixel[3] > threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    # Calculate crop coordinates
    crop_coords = (min_x, min_y, max_x + 1, max_y + 1)  # Add 1 to include the last pixel

    # Crop the image
    cropped_image = image.crop(crop_coords)
    cropped_image.save(output_image_path)

def extract_text_from_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        page_text = page.get_text("html")  
    return page_text

def parse_ecg_grid_rotated(html_content, debug=False):
    """
    Parse ECG data from HTML content that is rotated 90 degrees.
    In rotated format:
    - Same 'top' value = same COLUMN (vertical alignment)
    - Same 'left' value = same ROW (horizontal alignment)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract all text elements with their positions
    elements = []
    for p_tag in soup.find_all('p'):
        style = p_tag.get('style', '')
        span = p_tag.find('span')
        if span:
            text = span.get_text().strip()
            if text:
                top_match = re.search(r'top:([\d.]+)pt', style)
                left_match = re.search(r'left:([\d.]+)pt', style)
                
                if top_match and left_match:
                    top = float(top_match.group(1))
                    left = float(left_match.group(1))
                    elements.append({
                        'text': text,
                        'top': top,
                        'left': left
                    })
    
    # Sort by left (row), then by top (column within that row)
    elements.sort(key=lambda x: (x['left'], -x['top']))
    
    # Group by rows (elements with similar 'left' values)
    rows = []
    current_row = []
    current_left = None
    tolerance = 2  # pts tolerance for same row (horizontal alignment)
    
    for elem in elements:
        if current_left is None or abs(elem['left'] - current_left) <= tolerance:
            current_row.append(elem)
            if current_left is None:
                current_left = elem['left']
        else:
            if current_row:
                # Sort elements within the row by 'top' (vertical position = column order)
                current_row.sort(key=lambda x: -x['top'])
                rows.append(current_row)
            current_row = [elem]
            current_left = elem['left']
    
    if current_row:
        current_row.sort(key=lambda x: -x['top'])
        rows.append(current_row)
    
    # Display rows with their structure
    if debug:
        print("Rows found (left position groups):")
        print("=" * 80)
        for i, row in enumerate(rows):
            row_texts = [elem['text'] for elem in row]
            left_pos = row[0]['left']
            print(f"Row {i+1} (left={left_pos:.1f}pt): {' | '.join(row_texts)}")
    return rows

def extract_comprehensive_ecg_data(html_content):
    """
    Extract comprehensive ECG data including measurements, metadata, and comments.
    Returns a pandas DataFrame with one row containing all extracted fields.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract all text elements with their positions
    elements = []
    for p_tag in soup.find_all('p'):
        style = p_tag.get('style', '')
        span = p_tag.find('span')
        if span:
            text = span.get_text().strip()
            if text:
                top_match = re.search(r'top:([\d.]+)pt', style)
                left_match = re.search(r'left:([\d.]+)pt', style)
                
                if top_match and left_match:
                    top = float(top_match.group(1))
                    left = float(left_match.group(1))
                    elements.append({
                        'text': text,
                        'top': top,
                        'left': left
                    })
    
    # Sort by left (row), then by top in DECREASING order (column)
    elements.sort(key=lambda x: (x['left'], -x['top']))
    
    # Group by rows (same left position)
    rows = []
    current_row = []
    current_left = None
    tolerance = 2
    
    for elem in elements:
        if current_left is None or abs(elem['left'] - current_left) <= tolerance:
            current_row.append(elem)
            if current_left is None:
                current_left = elem['left']
        else:
            if current_row:
                current_row.sort(key=lambda x: -x['top'])
                rows.append(current_row)
            current_row = [elem]
            current_left = elem['left']
    
    if current_row:
        current_row.sort(key=lambda x: -x['top'])
        rows.append(current_row)
    
    # Initialize result dictionary
    ecg_dict = {
        'measurements': {},
        'metadata': {},
        'comments': {},
        'leads': {},
        'filter': None
    }

    # Erase the rows that are optional, not useful for us and start with fixed characters
    unwanted_rows = ("Technician","Test","Facility","Referred","Electronically")
    cleaned_rows = []
    for row in rows:
        if not row:
            continue
        first_text = row[0]['text']
        if not any (first_text.startswith(prefix) for prefix in unwanted_rows):
            cleaned_rows.append(row)
    rows = cleaned_rows  # Stay with the remaining rows only, avoiding uncertainty of which rows we will have

    first_row_with_10s_strip = 99 # When I detect the last row of the 2.5 strips, I will put a value to this
    # Extract data from specific rows
    for row_idx, row in enumerate(rows):
        row_texts = [elem['text'] for elem in row]
        row_string = ' '.join(row_texts)
        first_text =  row_texts[0] if row_texts else ""

        # Row 1: ID (2nd element), datetime (3rd element), comments (last elements)
        if row_idx == 0:
            if len(row_texts) >= 2:
                ecg_dict['metadata']['ID'] = row_texts[1]
            if len(row_texts) >= 3:
                ecg_dict['metadata']['datetime'] = row_texts[2]
        
        # Row 2: birthday (1st element), comments (last elements)
        elif row_idx == 1:
            if len(row_texts) >= 1:
                ecg_dict['metadata']['birthday'] = row_texts[0]
            if len(row_texts) >= 2:
                ecg_dict['comments']['row2'] = row_texts[-1]
        
        # Row 3: comments (last elements)
        elif row_idx == 2:
            if len(row_texts) >= 1:
                if not row_texts[-1].startswith("Electronically"):
                    ecg_dict['comments']['row3'] = row_texts[-1]
                ecg_dict['metadata']['gender'] = row_texts[0]    
        
        # Row 4: comments (last elements)
        elif row_idx == 3:
            if len(row_texts) >= 1:
                if not row_texts[-1].startswith("Electronically"):
                    ecg_dict['comments']['row4'] = row_texts[-1]
        
        # Row 5: comments (last elements)
        elif row_idx == 4:
            if len(row_texts) >= 1:
                if not row_texts[-1].startswith("Electronically"):
                    ecg_dict['comments']['row5'] = row_texts[-1]

        # Row 6 (when there are more rows of comments)        
        elif row_idx == 5 and not first_text.startswith ("I"):
            if len(row_texts) >=1:
                if not row_texts[-1].startswith("Electronically"):
                    ecg_dict['comments']['row6'] = row_texts[-1]

        # Rows directly below lead III: 10-second-leads
        if first_text.startswith("III"):
            first_row_with_10s_strip = row_idx + 1  #Next row will have 10s
        if row_idx == first_row_with_10s_strip:
            ecg_dict['leads']['10sLead1'] = row_texts
        elif (row_idx == first_row_with_10s_strip + 1):
            if "25mm/s" not in row_string:
                ecg_dict['leads']['10sLead2'] = row_texts
        elif (row_idx  == first_row_with_10s_strip + 2):
            if "Page" not in row_string:
                ecg_dict['leads']['10sLead3'] = row_texts
        
        # Row with filter value (3rd element in Hz)
        if first_text.startswith("25"):
            if len(row_texts) >= 3:
                filter_text = row_texts[2]
                # Extract numeric value from filter (e.g., "40Hz" -> 40)
                filter_match = re.search(r'(\d+\.?\d*)\s*Hz', filter_text)
                if filter_match:
                    ecg_dict['filter'] = float(filter_match.group(1))
                else:
                    ecg_dict['filter'] = filter_text
        
        # Extract measurements from any row
        # Ventricular rate
        if ('Vent. rate' in row_string or 'Vent.' in row_string) and 'BPM' in row_string:
            for text in row_texts:
                if text.isdigit() and 20 < int(text) < 300:
                    ecg_dict['measurements']['Vent_rate_bpm'] = int(text)
                    break
        
        # PR interval
        if 'PR interval' in row_string or 'PR' in row_string:
            for text in row_texts:
                if text.isdigit() and 80 < int(text) < 400:
                    ecg_dict['measurements']['PR_interval_ms'] = int(text)
                    break
        
        # QRS duration
        if 'QRS duration' in row_string:
            for text in row_texts:
                if text.isdigit() and 40 < int(text) < 200:
                    ecg_dict['measurements']['QRS_duration_ms'] = int(text)
                    break
        
        # QT/QTc
        if 'QT/QTc' in row_string or ('QT' in row_string and 'QTc' in row_string):
            for text in row_texts:
                qt_match = re.match(r'(\d+)/(\d+)', text)
                if qt_match:
                    ecg_dict['measurements']['QT_ms'] = int(qt_match.group(1))
                    ecg_dict['measurements']['QTc_ms'] = int(qt_match.group(2))
                    break
        # P-R-T axes
        if 'P-R-T' in row_string or 'axes' in row_string:
            values = []
            # Extract the part of the line after 'P-R-T Axes'
            after_axes = row_string.split('P-R-T', 1)[1]
            # Match numbers including decimals and negatives, or a literal asterisk
            matches = re.findall(r'-?\d+\.?\d*|\*', after_axes)
            for match in matches:
                if match == '*':
                    values.append(None)  # Use None for missing/unknown values
                else:
                    try:
                        val = float(match)
                        if abs(val) < 200:  # Valid axis range
                            values.append(val)
                    except:
                        pass
            if len(values) >= 3:
                ecg_dict['measurements']['P_axis'] = values[0]
                ecg_dict['measurements']['R_axis'] = values[1]
                ecg_dict['measurements']['T_axis'] = values[2]
    
    # Flatten the dictionary into a single-level dictionary for DataFrame
    flat_dict = {}
    
    # Add all measurements
    for key, value in ecg_dict.get('measurements', {}).items():
        flat_dict[key] = value
    
    # Add metadata
    for key, value in ecg_dict.get('metadata', {}).items():
        flat_dict[key] = value
    
    # Add comments
    all_comments = ' '.join(str (comment) for comment in ecg_dict.get('comments', {}).values() if comment)
    flat_dict[f'comments'] = all_comments
    
    # Add leads (convert lists to strings)
    for key, value in ecg_dict.get('leads', {}).items():
        if isinstance(value, list):
            flat_dict[f'leads_{key}'] = ', '.join(value)
        else:
            flat_dict[f'leads_{key}'] = value
    
    # Add filter
    if ecg_dict.get('filter'):
        flat_dict['filter_Hz'] = ecg_dict['filter']
    
    # Convert to DataFrame with single row
    df = pd.DataFrame([flat_dict])
    
    return df
def parse_ecg_features(html_content):
    """
    Parse ECG features from HTML content.
    Groups text elements by their vertical position (top value) to identify features on the same line.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract all text elements with their positions
    text_elements = []
    for p_tag in soup.find_all('p'):
        style = p_tag.get('style', '')
        span = p_tag.find('span')
        if span:
            text = span.get_text().strip()
            if text:
                # Extract top and left positions
                top_match = re.search(r'top:([\d.]+)pt', style)
                left_match = re.search(r'left:([\d.]+)pt', style)
                
                if top_match and left_match:
                    top = float(top_match.group(1))
                    left = float(left_match.group(1))
                    text_elements.append({
                        'text': text,
                        'top': top,
                        'left': left
                    })
    
    # Sort by top position, then by left position
    text_elements.sort(key=lambda x: (x['top'], x['left']))
    
    # Group elements that are on the same line (within 1pt tolerance)
    lines = []
    current_line = []
    current_top = None
    
    for elem in text_elements:
        if current_top is None or abs(elem['top'] - current_top) < 1:
            current_line.append(elem)
            current_top = elem['top']
        else:
            if current_line:
                lines.append(current_line)
            current_line = [elem]
            current_top = elem['top']
    
    if current_line:
        lines.append(current_line)
    
    # Combine text from each line
    combined_lines = []
    for line in lines:
        line_text = ' '.join([elem['text'] for elem in line])
        combined_lines.append(line_text)
    
    return combined_lines

def parse_text_ecg_report(lines):
    """
    Parse ECG report from text lines and return a structured DataFrame.
    
    Args:
        lines (list): List of text lines from the ECG report
    Returns:
        pd.DataFrame: DataFrame containing parsed ECG data
    """
    ecg_dict = {
        'hospital': '',
        'address': [],
        'metadata': {},
        'measurements': {},
        'comments': []
    }
    findings_in_next_line = False
    # Parse lines
    for i, line in enumerate(lines):
        line = line.strip()
        # Patient metadata
        if 'Patient Name:' in line:
            parts = line.split()
            if len(parts) >= 4:
                if 'MR #:' in line:
                    mr_index = line.find('MR #:')
                    if mr_index != -1:
                        ecg_dict['metadata']['ID'] = line[mr_index:].split()[-1]
        
        # Patient age/gender/date
        elif line.count('/') >= 2 and "Acct" in line:
            parts = line.split('/')
            if len(parts) >= 3:
                ecg_dict['metadata']['birthday'] = parts[0].strip() + '/' + parts[1].strip() +'/' + parts[2].strip()  #To conatenate MM/DD/YYYY
                ecg_dict['metadata']['age'] = parts[3].strip().split()[0]
                remaining = parts[-1].strip()
                if remaining.startswith("F"):
                    ecg_dict['metadata']['gender'] = 'Female' 
                elif remaining.startswith("M"):
                    ecg_dict['metadata']['gender'] = 'Male' 
        
        
        # Measurements
        elif 'Vent. Rate :' in line:
            matches = re.findall(r'(\d+)\s*BPM', line)
            if matches:
                ecg_dict['measurements']['Vent_rate_bpm'] = int(matches[0])
            matches = re.findall(r'Atrial Rate\s*:\s*(\d+)\s*BPM', line)
            if matches:
                ecg_dict['measurements']['atrial_rate_bpm'] = int(matches[0])
        
        elif 'PR Int' in line or 'P-R Int' in line:
            matches = re.findall(r':\s*(\d+)\s*ms', line)
            if matches:
                ecg_dict['measurements']['PR_interval_ms'] = int(matches[0])
        
        if 'QRS Dur' in line:
            after_qrs = line.split("QRS Dur", 1)[1]
            matches = re.findall(r':\s*(\d+)\s*ms', after_qrs)
            if matches:
                ecg_dict['measurements']['QRS_duration_ms'] = int(matches[0])
        
        elif 'QT Int' in line:
            after_qtint = line.split("QT Int", 1)[1]
            matches = re.findall(r':\s*(\d+)\s*ms', after_qtint)
            if matches:
                ecg_dict['measurements']['QT_ms'] = int(matches[0])
        
        elif 'QTc Int' in line:
            matches = re.findall(r':\s*(\d+)\s*ms', line)
            if matches:
                ecg_dict['measurements']['QTc_ms'] = int(matches[0])
                index_of_comments = i + 1
                findings_in_next_line = True
                continue
        
        if 'P-R-T Axes' in line:
            matches = re.findall(r':\s*([-\d]+|\*)\s*([-\d]+|\*)\s*([-\d]+|\*)\s*degrees', line)
            if matches and len(matches[0]) == 3:
                def parse_axis(value):
                    return int(value) if value != '*' else None

                ecg_dict['measurements']['P_axis'] = parse_axis(matches[0][0])
                ecg_dict['measurements']['R_axis'] = parse_axis(matches[0][1])
                ecg_dict['measurements']['T_axis'] = parse_axis(matches[0][2])

       
        # Looking for the datetime timestamp
        if "Confirmed by" in line:
            line_split = line.split("on")
            ecg_dict['metadata']['datetime'] = line_split[-1]
    
        # Interpretation/Findings
        if findings_in_next_line:
            if 'Confirmed by' not in line:
                ecg_dict['comments'].append(line)
            else:
                findings_in_next_line =  False  
        

    # Flatten dictionary for DataFrame
    flat_dict = {}

    
    # Add metadata
    for key, value in ecg_dict['metadata'].items():
        flat_dict[key] = value
    
    # Add measurements
    for key, value in ecg_dict['measurements'].items():
        flat_dict[key] = value
    
    # Add findings
    flat_dict['comments'] = ' | '.join(ecg_dict['comments'])
    
    # Convert to DataFrame
    df = pd.DataFrame([flat_dict])
    return df

# # Parse the HTML
# lines = parse_ecg_features(extracted_text)  
# test_df = parse_text_ecg_report(lines)
# print("Extracted lines:")
# for i, line in enumerate(lines, 1):
#     print(f"{i}. {line}")

# test_df.head()

# Define the function to process a single file
def process_pdf(filename, error_log_filename, pdf_file_path, output_dir):
    try:
        extracted_text = extract_text_from_pdf(os.path.join(pdf_file_path, filename))
        if "NEURODIAGNOSTIC REPORT" in extracted_text:
            lines_of_text = parse_ecg_features(extracted_text)
            df = parse_text_ecg_report(lines_of_text)
            df['source_file'] = filename
        else:
            extract_vectors_from_pdf(os.path.join(pdf_file_path, filename), output_dir)
            df = extract_comprehensive_ecg_data(extracted_text)
            df['source_file'] = filename
        return df
    except Exception as e:
        error_message = f"Error processing {filename}: {str(e)}\n"
        print(error_message)

        # Append error to log file
        with open(error_log_filename, "a") as log_file:
            log_file.write(error_message)
        return None

def main(args):
    
    print(f"output_dir: {args.output_dir}")
    print(f"pdf_folder: {args.pdf_file_path}")
    print(f"filename_prefix: {args.filename_prefix}")
    print(f"Will process indices from {args.start_index} to {args.end_index}")
 
    pdf_file_path = args.pdf_file_path
    output_dir = args.output_dir 
    filename_prefix = args.filename_prefix
    start_index = args.start_index
    end_index = args.end_index


    file_list = os.listdir(pdf_file_path)
    # Limit to first 10 files for testing
    # First check if end_index -99, if so, process until last file
    if end_index == -99:
        file_list_subset = file_list[start_index:]
    else:
        file_list_subset = file_list[start_index:end_index]


    ecg_dfs = []
    start_time = time.time()

    #Setup logging
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_log_filename = f"{filename_prefix}error_log_{timestamp_str}.txt"
    log_header = (
        f"\n=== Error Log Started ===\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total files to process: {len(file_list_subset)}\n"
        f"Folder to process: {pdf_file_path}\n"
        f"Output folder: {output_dir}\n"
        f"First file: {file_list_subset[0]}\n"
        f"Last file: {file_list_subset[-1]}\n"
        f"==========================\n\n"
    )
    with open(error_log_filename, "w") as log_file:
        log_file.write(log_header)

    # Use multiprocessing to process files in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, filename, error_log_filename, pdf_file_path, output_dir): filename for filename in file_list_subset}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                ecg_dfs.append(result)

            # Progress update every 1000 files
            if i % 1000 == 0 or i == len(futures):
                elapsed = time.time() - start_time
                print(f"Processed {i}/{len(futures)} files in {elapsed:.1f} seconds")

    end_time = time.time()

    # Concatenate all dataframes
    combined_ecg_df = pd.concat(ecg_dfs, ignore_index=True)
    ending_message = f"\nFinal dataframe shape: {combined_ecg_df.shape} \nTotal processing time: {end_time - start_time:.2f} seconds"
    # Save the combined dataframe


    print(ending_message)
    with open(error_log_filename, "a") as log_file:
        log_file.write(ending_message)
    csv_filename = f"{filename_prefix}combined_ecg_data_{timestamp_str}.csv"
    combined_ecg_df.to_csv(csv_filename, index=False)



import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to SVG conversion.")
    parser.add_argument("--output_dir", type=str, default="/condo/alkindilab/hmaiaxg21/ecg_pdf_output/20xx", help="Output folder")
    parser.add_argument("--pdf_file_path", type=str, default="/condo/alkindilab/shared/ECG_Images/OnBaseECG_Martin/2019", help="Folder where the pdfs to convert are")
    parser.add_argument("--filename_prefix", type=str, default="year_xx_", help="Filename to save the numpy array of 12 leads")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to process, from file list in svg folder")
    parser.add_argument("--end_index", type=int, default=100000, help="End index to process, from file list in svg folder")
    args = parser.parse_args()
    main(args)

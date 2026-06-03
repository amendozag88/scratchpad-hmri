#Helper functions
import matplotlib.pyplot as plt
import numpy as np

#Create a function that accepts the calibration signals and returns the X coordinate at origin and at 1
def get_calibration_coordinates(paths_calibration, debug = False):
    #Iterate through the paths, until we find a path with length > 1
    for j in range(0,len(paths_calibration)-1):
        #Get the length of the path. The calibration signal is a big jump
        length = paths_calibration[j].length()
        if length > 1:
            #Save the current X as x-start-calibration-signal
            x_origin = paths_calibration[j].start.real
            x_unit = paths_calibration[j].end.real
            y_start_first_line = paths_calibration[j].start.imag
            y_end_first_line = paths_calibration[j].end.imag
            length_first_line = length
            if (debug):
                print(f"Length of path {j} is {length_first_line}. X axis (magnitude)")
                print(f"Resolution of magnitude is {1/length_first_line}")
                print(f"X coordinate at origin: {x_origin}")
                print(f"X coordinate at unit magnitude: {x_unit}")
            j+=1
            break

        #Get the second encountered big lenght, which belongs to the second line, y-axis
    for j in range (j,len(paths_calibration)-1):
        length = paths_calibration[j].length()
        if length > 1:
            y_start_second_line = paths_calibration[j].start.imag
            y_end_second_line = paths_calibration[j].end.imag
            if (debug):
                print(f"Length of path {j} is {length_first_line}. X axis (magnitude) third line")
            break
    y_length = y_end_first_line - y_start_second_line 
    if (debug):
        print(f"Length of second line y-axis (time): {y_length}")
    return x_origin, x_unit, y_length

#Function to get the lead signal
def get_lead_signal (paths_lead, x_origin, x_unit, original_unit_or_mv = 'original', plot_signal = False):
    lead_signal = np.zeros(len(paths_lead))

    for i, path in enumerate (paths_lead):
        #lead_signal[i] = path.start.real - x_origin  #It was inverted I think
        lead_signal[i] = x_origin - path.start.real  #Trying it like this

    if original_unit_or_mv == 'mv':
        # I would just scale all values by a factor
        #x_origin_1 - x_unit_1 #This is 1 mv according to the standard calibration signal
        lead_signal = lead_signal * 1/(x_origin - x_unit)
    if (plot_signal):
        plt.figure(figsize=(20,5))
        plt.plot(lead_signal)
    return lead_signal

def get_all_lead_signals (paths, intervals_start, intervals_end, 
                          x_origin_1, x_origin_2, x_origin_3, 
                          x_unit_1, x_unit_2, x_unit_3, 
                          original_unit_or_mv = 'original', plot_signal = False):
    #TODO refactor this
    paths_lead_1 = paths[intervals_start[3]:intervals_end[3]]
    paths_lead_2 = paths[intervals_start[4]:intervals_end[4]]
    paths_lead_3 = paths[intervals_start[5]:intervals_end[5]]
    paths_lead_4 = paths[intervals_start[6]:intervals_end[6]]
    paths_lead_5 = paths[intervals_start[7]:intervals_end[7]]
    paths_lead_6 = paths[intervals_start[8]:intervals_end[8]]
    paths_lead_7 = paths[intervals_start[9]:intervals_end[9]]
    paths_lead_8 = paths[intervals_start[10]:intervals_end[10]]
    paths_lead_9 = paths[intervals_start[11]:intervals_end[11]]
    paths_lead_10 = paths[intervals_start[12]:intervals_end[12]]
    paths_lead_11 = paths[intervals_start[13]:intervals_end[13]]
    paths_lead_12 = paths[intervals_start[14]:intervals_end[14]]
    
    #Create the array that will store all leads
    ECG = np.zeros((len(paths_lead_1),12))

    #Get the lead signals, mapping them to the appropriate origin
    ECG[:,0] = get_lead_signal(paths_lead_1, x_origin_1, x_unit_1, original_unit_or_mv, plot_signal)
    ECG[:,1] = get_lead_signal(paths_lead_2, x_origin_2, x_unit_2, original_unit_or_mv, plot_signal)
    ECG[:,2] = get_lead_signal(paths_lead_3, x_origin_3, x_unit_3, original_unit_or_mv, plot_signal)
    ECG[:,3] = get_lead_signal(paths_lead_4, x_origin_1, x_unit_1, original_unit_or_mv, plot_signal)
    ECG[:,4] = get_lead_signal(paths_lead_5, x_origin_2, x_unit_2, original_unit_or_mv, plot_signal)
    ECG[:,5] = get_lead_signal(paths_lead_6, x_origin_3, x_unit_3, original_unit_or_mv, plot_signal)
    ECG[:,6] = get_lead_signal(paths_lead_7, x_origin_1, x_unit_1, original_unit_or_mv, plot_signal)
    ECG[:,7] = get_lead_signal(paths_lead_8, x_origin_2, x_unit_2, original_unit_or_mv, plot_signal)
    ECG[:,8] = get_lead_signal(paths_lead_9, x_origin_3, x_unit_3, original_unit_or_mv, plot_signal)
    ECG[:,9] = get_lead_signal(paths_lead_10, x_origin_1, x_unit_1, original_unit_or_mv, plot_signal)
    ECG[:,10] = get_lead_signal(paths_lead_11, x_origin_2, x_unit_2, original_unit_or_mv, plot_signal)
    ECG[:,11] = get_lead_signal(paths_lead_12, x_origin_3, x_unit_3, original_unit_or_mv, plot_signal)
    
    return ECG

def get_rythym_strips_signals (paths, intervals_start, intervals_end, 
                          x_origin_4, x_origin_5, x_origin_6, 
                          x_unit_4, x_unit_5, x_unit_6, 
                          original_unit_or_mv = 'original', get_3_10s_leads = True, plot_signal = False):

    #TODO Here put a check, certain ECGs have only 1 rhythym strip
    
    if get_3_10s_leads:
        paths_10s_1 = paths[intervals_start[18]:intervals_end[18]]
        ECG_10s = np.zeros((len(paths_10s_1),3))
        paths_10s_2 = paths[intervals_start[19]:intervals_end[19]]
        paths_10s_3 = paths[intervals_start[20]:intervals_end[20]]
        ECG_10s[:,1] = get_lead_signal(paths_10s_2, x_origin_5, x_unit_5, original_unit_or_mv, plot_signal)
        ECG_10s[:,2] = get_lead_signal(paths_10s_3, x_origin_6, x_unit_6, original_unit_or_mv, plot_signal)
    else:
        paths_10s_1 = paths[intervals_start[16]:intervals_end[16]]
        ECG_10s = np.zeros((len(paths_10s_1),3))
    
    ECG_10s[:,0] = get_lead_signal(paths_10s_1, x_origin_4, x_unit_4, original_unit_or_mv, plot_signal)
    
    return ECG_10s

#//////////////////////

def process_svg_file(filename, svg_file_path_or_bytes, from_bytes=False):
    try:
        # Load SVG content
        if from_bytes:
            svg_stream = BytesIO(svg_file_path_or_bytes)
            paths, attributes, svg_attributes = svg2paths2(svg_stream)
        else:
            full_path = os.path.join(svg_file_path_or_bytes, filename)
            paths, attributes, svg_attributes = svg2paths2(full_path)
        # Detect continuity intervals
        intervals_start = [0]
        intervals_end = []
        for j in range(len(paths) - 1):
            if (paths[j].end.real != paths[j + 1].start.real) or (paths[j].end.imag != paths[j + 1].start.imag):
                intervals_end.append(j)
                intervals_start.append(j + 1)
        intervals_end.append(len(paths) - 1)
        get_3_10s_leads = False

        if len(intervals_start) > 17:  #IF 17 intervals, we have only 1 10seconds row, if 20 intervals, we have all
            get_3_10s_leads = True
        
        #Lets retrieve the paths for the first calibration signal, with info about the origins to get the signals later
        debug_calibration = False
        x_origin_1, x_unit_1, y_length_1 = get_calibration_coordinates(paths[intervals_start[0]:intervals_end[0]], debug_calibration) #First calibration signal
        x_origin_2, x_unit_2, y_length_2 = get_calibration_coordinates(paths[intervals_start[1]:intervals_end[1]], debug_calibration) #second calibration signal
        x_origin_3, x_unit_3, y_length_3 = get_calibration_coordinates(paths[intervals_start[2]:intervals_end[2]], debug_calibration) #third calibration signal
        x_origin_4, x_unit_4, y_length_4 = get_calibration_coordinates(paths[intervals_start[15]:intervals_end[15]], debug_calibration) #fourth calibration signal
        x_origin_5 = x_unit_5 = y_length_5 = x_origin_6 = x_unit_6 = y_length_6 = 0 #In case there are no 3 10 second leads
        if get_3_10s_leads:
            x_origin_5, x_unit_5, y_length_5 = get_calibration_coordinates(paths[intervals_start[16]:intervals_end[16]], debug_calibration) #fifth calibration signal
            x_origin_6, x_unit_6, y_length_6 = get_calibration_coordinates(paths[intervals_start[17]:intervals_end[17]], debug_calibration) #sixth calibration signal


        # Extract signals
        ECG_12_lead_mV = get_all_lead_signals(paths, intervals_start, intervals_end,
                                              x_origin_1, x_origin_2, x_origin_3,
                                              x_unit_1, x_unit_2, x_unit_3,
                                              'mv', False)

        ECG_10s_raw_mV = get_rythym_strips_signals(paths, intervals_start, intervals_end,
                                                   x_origin_4, x_origin_5, x_origin_6,
                                                   x_unit_4, x_unit_5, x_unit_6,
                                                   'mv', get_3_10s_leads, #Flag to check if we get 3 10 seconds lead or just 1
                                                    False) #The final true or false is for plotting

        return np.array(ECG_12_lead_mV), np.array(ECG_10s_raw_mV)

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return None, None

def run_ecg_pipeline(file_list, svg_file_path, output_12_leads_path, output_10s_strips_path, timestamp):
    try:
        start_time = datetime.now()
        logging.info(f"Started ECG processing for {len(file_list)} files")

        all_ECG_12_leads = []
        all_ECG_10s_strips = []
        successful_filenames = []

        with ProcessPoolExecutor() as executor:
            ordered_futures = []
            for filename in file_list:
                if filename.endswith('.svg'):
                    future = executor.submit(process_svg_file, filename, svg_file_path, False) #False bc we want to open svg files themselves
                    ordered_futures.append((filename, future))
                elif filename.endswith('.svg.gz'):
                #Uncompress it first in a temp file
                    try:
                        gz_path = os.path.join(svg_file_path, filename)
                        with gzip.open(gz_path, 'rb') as f_in:
                            svg_bytes = f_in.read()

                        # Submit in-memory SVG bytes
                        future = executor.submit(process_svg_file, filename, svg_bytes, True) #True bc we are sending an in-memory svg-like thing
                        ordered_futures.append((filename, future))

                    except Exception as e:
                        logging.error(f"Error decompressing {filename}: {e}")

            for i, (filename, future) in enumerate(ordered_futures):
                result_12, result_10s = future.result()

                if result_12 is not None and result_10s is not None:
                    all_ECG_12_leads.append(result_12)
                    all_ECG_10s_strips.append(result_10s)
                    successful_filenames.append(filename)

                if (i + 1) % 1000 == 0:
                    logging.info(f"Processed {i + 1} files")
                    print(f"Processed {i + 1} files")

        # Convert to 3D arrays
        ecg_12_array = np.stack(all_ECG_12_leads, axis=0)
        ecg_10s_array = np.stack(all_ECG_10s_strips, axis=0)

        # Save arrays
        np.save(output_12_leads_path, ecg_12_array)
        np.save(output_10s_strips_path, ecg_10s_array)
        np.save(f'successful_filenames_{timestamp}.npy', np.array(successful_filenames))

        end_time = datetime.now()
        logging.info(f"Finished ECG processing. Duration: {end_time - start_time}")
        logging.info(f"Saved 12-lead ECGs with shape {ecg_12_array.shape} to {output_12_leads_path}")
        logging.info(f"Saved 10-second strips with shape {ecg_10s_array.shape} to {output_10s_strips_path}")
        logging.info(f"Saved {len(successful_filenames)} successful filenames to successful_filenames.npy")

    except Exception as e:
        logging.error(f"Fatal error in pipeline: {e}")



##?/////////////////////////////////////////////////////////////////////////////
import os
import numpy as np
from svgpathtools import svg2paths2

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
import logging
#Adding support for the .svg.gz type of files
import gzip
from io import BytesIO

def main(args):
    print(f"output_dir: {args.output_dir}")
    print(f"svg_folder: {args.svg_folder}")
    print(f"filename_12_leads_numpy: {args.filename_12_leads_numpy}")
    print(f"filename_10s_strips_numpy: {args.filename_10s_strips_numpy}")
    print(f"Will process indices from {args.start_index} to {args.end_index}")

    svg_file_path = args.svg_folder
    output_dir = args.output_dir 
    filename_12_leads_numpy = args.filename_12_leads_numpy
    filename_10s_strips_numpy  = args.filename_10s_strips_numpy
    start_index = args.start_index
    end_index = args.end_index

    # ------- TESTING WITH A FIXED SINGLE FILE ----------------------------

    # filename = "MRN-000075937_Order-_DocID-12645658_PageNum-1_p0.svg"  # Example SVG filename, just to check the code works

    # paths, attributes, svg_attributes = svg2paths2(filename)
    # #Let's check for continuity intervals in the signal 
    # intervals_start = []
    # intervals_end = []

    # #iterate through all paths
    # intervals_start.append(0)
    # for j in range(0,len(paths)-1):
    # #Continuity is when path x[j].end.real and path x[j+1].start.real are the same, and the same for imaginary
    #     if (paths[j].end.real == paths[j+1].start.real) and (paths[j].end.imag == paths[j+1].start.imag):
    #         pass #continuity
    #     else:
    #         #print(f"Discontinuity between {j} and {j+1}")
    #         intervals_end.append(j)
    #         intervals_start.append(j+1)
    # intervals_end.append(len(paths)-1)
    
    # get_3_10s_leads = False

    # if len(intervals_start) > 17:  #IF 17 intervals, we have only 1 10seconds row, if 20 intervals, we have all
    #     get_3_10s_leads = True
    # print(f"number of intervals in test ECG: {len(intervals_start)}")
    # #Lets retrieve the paths for the first calibration signal, with info about the origins to get the signals later
    # debug_calibration = False
    # x_origin_1, x_unit_1, y_length_1 = get_calibration_coordinates(paths[intervals_start[0]:intervals_end[0]], debug_calibration) #First calibration signal
    # x_origin_2, x_unit_2, y_length_2 = get_calibration_coordinates(paths[intervals_start[1]:intervals_end[1]], debug_calibration) #second calibration signal
    # x_origin_3, x_unit_3, y_length_3 = get_calibration_coordinates(paths[intervals_start[2]:intervals_end[2]], debug_calibration) #third calibration signal
    # x_origin_4, x_unit_4, y_length_4 = get_calibration_coordinates(paths[intervals_start[15]:intervals_end[15]], debug_calibration) #fourth calibration signal
    # x_origin_5 = x_unit_5 = y_length_5 = x_origin_6 = x_unit_6 = y_length_6 = 0 #In case there are no 3 10 second leads
    # if get_3_10s_leads:
    #     x_origin_5, x_unit_5, y_length_5 = get_calibration_coordinates(paths[intervals_start[16]:intervals_end[16]], debug_calibration) #fifth calibration signal
    #     x_origin_6, x_unit_6, y_length_6 = get_calibration_coordinates(paths[intervals_start[17]:intervals_end[17]], debug_calibration) #sixth calibration signal
    # print(f"x_origins and units of test ECG: {x_origin_1}, {x_origin_2}, {x_origin_3}, {x_origin_4}, {x_origin_5}, {x_origin_6}")
    # print(f"{x_unit_1}, {x_unit_2}, {x_unit_3}, {x_unit_4}, {x_unit_5}, {x_unit_6}")    
    # ECG_12_lead_mV = get_all_lead_signals(paths, intervals_start, intervals_end,x_origin_1, x_origin_2, x_origin_3,x_unit_1, x_unit_2, x_unit_3,'mv', False ) #The final true or false is for plotting

    # ECG_10s_raw_mV = get_rythym_strips_signals(paths, intervals_start, intervals_end,x_origin_4, x_origin_5, x_origin_6,x_unit_4, x_unit_5, x_unit_6,
    #                                     'mv', get_3_10s_leads, #Flag to check if we get 3 10 seconds lead or just 1
    #                                         True) #The final true or false is for plotting

    ############///////////////////////////////////////////

    #svg_file_path = "/condo/alkindilab/shared/ECG_Images/OnBaseECG_Martin/2016"  #Not used here
    #svg_file_path = "/condo/alkindilab/hmaiaxg21/ecg_pdf_output/2019/svg/"
    file_list = os.listdir(svg_file_path)

    # Define output paths
    #output_dir = "/condo/alkindilab/hmaiaxg21/ecg_pdf_output/"
    #output_12_leads_path = os.path.join(output_dir, 'all_ECG_12_leads.npy')
    #output_10s_strips_path = os.path.join(output_dir, 'all_ECG_10s_strips.npy')
    output_12_leads_path = os.path.join(output_dir, filename_12_leads_numpy)
    output_10s_strips_path = os.path.join(output_dir, filename_10s_strips_numpy)
    os.makedirs(output_dir, exist_ok=True)
    all_ECG_12_leads = []
    all_ECG_10s_strips = []



    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'ecg_processing_{timestamp}.log'  # timestamped log filename
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    #Check for end_index, if it is -99, that means we wanted to take files until the end
    if end_index == -99:
        run_ecg_pipeline(file_list[start_index:], svg_file_path, output_12_leads_path, output_10s_strips_path, timestamp)
    else:
        run_ecg_pipeline(file_list[start_index:end_index], svg_file_path, output_12_leads_path, output_10s_strips_path, timestamp)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG to numpy processing.")
    parser.add_argument("--output_dir", type=str, default="/condo/alkindilab/hmaiaxg21/ecg_pdf_output/", help="Output folder")
    parser.add_argument("--svg_folder", type=str, default="/condo/alkindilab/hmaiaxg21/ecg_pdf_output/2019/svg/", help="Folder where the svgs are")
    parser.add_argument("--filename_12_leads_numpy", type=str, default="all_ECG_12_leads.npy", help="Filename to save the numpy array of 12 leads")
    parser.add_argument("--filename_10s_strips_numpy", type=str, default="all_ECG_10s_strips.npy", help="Filename to save the numpy array of 10s leads")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to process, from file list in svg folder")
    parser.add_argument("--end_index", type=int, default=100000, help="End index to process, from file list in svg folder")
    args = parser.parse_args()
    main(args)


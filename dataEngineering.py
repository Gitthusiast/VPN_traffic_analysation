
import csv
import os
import pandas as pd
import numpy as np

browsing_type = {
    'Skype_chat': 'Chat',
    'Facebook': 'Chat',
    'GoogleHangouts': 'Chat',
    'Whatsapp_chat': 'Chat',
    'Telegram_chat': 'Chat',

    'Youtube': 'Streaming',
    'Netflix': 'Streaming',
    'Vimeo': 'Streaming',

    'qBitTorrent': 'qBittorrent',
    'qBittorrent': 'qBittorrent',

    'Random_Websites': 'Random_Websites',

    'Skype_files': 'File Transfer',
    'Dropbox': 'File Transfer',
    'gdrive': 'File Transfer',
    'Whatsapp_files': 'File Transfer',
    'Telegram_files': 'File Transfer',

    'Skype_video': 'Video Conferencing',
    'GoogleMeets': 'Video Conferencing',
    'Zoom': 'Video Conferencing',
    'Microsoft_teams': 'Video Conferencing',
    'MicrosoftTeams': 'Video Conferencing'
}

TIME_EPOCH = 0
FRAME_LEN = 1
SRC_PORT = 2

IN_PACKET = 1
OUT_PACKET = 0

def data_preprocessing():

    dirs = ['.\\place1\\England']  # , '.\\place2\\Japan']
    for directory in dirs:

        # write all data into output labeled file
        with open(f'.\\labeled_{directory[9:]}.tsv', 'wt', newline='') as out_file:
            time = 0.0
            prev_time = 0.0
            tsv_writer = csv.writer(out_file, delimiter='\t')

            tsv_writer.writerow(['frame.time_epoch', 'frame.len', 'tcp.srcport',
                                 'Random website', 'Browsing', 'Country', 'in/out', 'time_delta'])

            for filename in os.listdir(directory):
                with open(directory + '\\' + filename) as file:
                    tsv_file = csv.reader(file, delimiter="\t")
                    app_name = filename.split('_')
                    # extract label from filename
                    if not 'VPN' in app_name:
                        label = '_'.join(app_name[:app_name.index('NONVPN')])
                    else:
                        label = '_'.join(app_name[:app_name.index('VPN')])
                    print(label)

                    for i, line in enumerate(tsv_file):
                        if i == 0:  # skip heading
                            continue
                        prev_time = time
                        time = float(line[0])
                        in_out = OUT_PACKET if int(line[SRC_PORT]) >= 49151 else IN_PACKET
                        # count delta frame.time between consecutive packets
                        if i == 1:
                            tsv_writer.writerow([line[TIME_EPOCH], line[FRAME_LEN], line[SRC_PORT], label,
                                                 browsing_type[label], directory[9:], in_out, 0])
                        else:
                            tsv_writer.writerow([line[TIME_EPOCH], line[FRAME_LEN], line[SRC_PORT], label,
                                                 browsing_type[label], directory[9:], in_out, time-prev_time])


def feature_aggregation(init_df):

    last_row = int(init_df.shape[0]) + 1 if int(init_df.shape[0])//10 >0 else 0
    stream_key = [int(i/10) for i in range(init_df.shape[0])]
    init_df = init_df.assign(stream_key=stream_key)

    init_df = count_delta_time_average_and_std_per_10_packets(init_df, last_row)
    # init_df = average_and_std_packet_len_per_10_packets(init_df)
    # init_df.dropna(inplace=True)

    return init_df


def count_delta_time_average_and_std_per_10_packets(init_df, last_row):

    """counter = 1
    dt_vals = []

    for row in range(init_df.shape[0]):

        if counter > 10:
            for i in range(10):
                init_df.loc[init_df.index[row-i], 'delta_time_average'] = np.average(dt_vals)
                init_df.loc[init_df.index[row-i], 'delta_time_std'] = np.std(dt_vals)
            counter = 1
            dt_vals = []
        counter += 1
        dt_vals.append(float(init_df.loc[init_df.index[row], 'time_delta']))"""

    average_values_dict = {}
    std_values_dict = {}

    for i_stream in range(last_row):
        mean = init_df.loc[init_df['stream_key'] == i_stream]['time_delta'].mean()
        std = init_df.loc[init_df['stream_key'] == i_stream]['time_delta'].std()
        average_values_dict[i_stream] = mean
        std_values_dict[i_stream] = std
    init_df['average_delta_time'] = init_df['stream_key'].apply(set_row_feature, args=(average_values_dict,))
    init_df['std_delta_time'] = init_df['stream_key'].apply(set_row_feature, args=(std_values_dict,))

    return init_df


def average_and_std_packet_len_per_10_packets(init_df):

    average_values_dict = {}
    std_values_dict = {}

    for i_stream in sorted(init_df['stream_key'].unique()):
        mean = init_df.loc[init_df['stream_key'] == i_stream]['tcp.len'].mean()
        std = init_df.loc[init_df['stream_key'] == i_stream]['tcp.len'].std()
        average_values_dict[i_stream] = mean
        std_values_dict[i_stream] = std
    init_df['average_len'] = init_df['stream_key'].apply(set_row_feature, args=(average_values_dict,))
    init_df['std_len'] = init_df['stream_key'].apply(set_row_feature, args=(std_values_dict,))

    return init_df


def set_row_feature(row_value, values_dict):
    """
        This is a helper function for the dataframe's apply method
    """
    return values_dict[row_value]


if __name__ == '__main__':

    data_preprocessing()
    train_df = pd.read_csv('.\\labeled_England.tsv', sep='\t')
    feature_aggregation(train_df)
    train_df.to_csv('.\\final_labeled_England.tsv', sep='\t', index=False)
    print('fin')

import csv
import os
import pandas as pd

browsing_type = {
    'Random_websites': 'Browsing',

    'Skype_chat': 'Chat',
    'Facebook': 'Chat',
    'GoogleHangouts': 'Chat',
    'Whatsapp_chat': 'Chat',
    'Telegram_chat': 'Chat',

    'Youtube': 'Streaming',
    'Netflix': 'Streaming',
    'Vimeo': 'Streaming',

    'qBitTorrent': 'File Transfer',
    'qBittorrent': 'File Transfer',
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

double_alias = {'Microsoft_teams': 'Microsoft_teams',
                'MicrosoftTeams': 'Microsoft_teams',
                'qBitTorrent': 'qBitTorrent',
                'qBittorrent': 'qBitTorrent'}

TIME_EPOCH = 0
FRAME_LEN = 1
SRC_PORT = 2

IN_PACKET = 1
OUT_PACKET = 0


def data_preprocessing():
    """
    This function creates labeled + time delta field added files for each of the given tsv files
    The aggregated files are saved into place1_labeled and place2_labeled directories
    """
    dirs = ['.\\place1\\England', '.\\place2\\Japan']
    for directory in dirs:

        for filename in os.listdir(directory):

            port_dt_dict_jap = {}
            port_dt_dict_eng = {}

            with open(directory + '\\' + filename) as file:
                # write all data into output labeled file
                if directory == dirs[0]:
                    out_filename = f'.\\place1_labeled\\labeled_{filename}'
                else:
                    out_filename = f'.\\place2_labeled\\labeled_{filename}'
                with open(out_filename, 'wt', newline='') as out_file:
                    time = 0.0
                    tsv_writer = csv.writer(out_file, delimiter='\t')

                    tsv_writer.writerow(['frame.time_epoch', 'frame.len', 'tcp.srcport',
                                         'applications_and_websites', 'categories', 'Country', 'io_packet', 'time_delta'])
                    tsv_file = csv.reader(file, delimiter="\t")
                    app_name = filename.split('_')
                    # extract label from filename
                    if not 'VPN' in app_name:
                        label = '_'.join(app_name[:app_name.index('NONVPN')])
                    else:
                        label = '_'.join(app_name[:app_name.index('VPN')])
                    print(label)
                    if label in double_alias.keys():
                        label = double_alias[label]

                    for i, line in enumerate(tsv_file):
                        if i == 0:  # skip heading
                            continue

                        prev_time = time
                        time = float(line[0])
                        in_out = OUT_PACKET if int(line[SRC_PORT]) >= 49151 else IN_PACKET
                        # count delta frame.time between consecutive packets
                        if directory == dirs[0] and line[SRC_PORT] not in port_dt_dict_eng.keys() \
                                or directory == dirs[1] and line[SRC_PORT] not in port_dt_dict_jap.keys():
                            tsv_writer.writerow([line[TIME_EPOCH], line[FRAME_LEN], line[SRC_PORT], label,
                                                 browsing_type[label], directory[9:], in_out, 0])
                        else:
                            if directory == dirs[0]:
                                tsv_writer.writerow([line[TIME_EPOCH], line[FRAME_LEN], line[SRC_PORT], label,
                                                     browsing_type[label], directory[9:], in_out,
                                                     float(line[0]) - port_dt_dict_eng[line[SRC_PORT]]])
                            else:
                                tsv_writer.writerow([line[TIME_EPOCH], line[FRAME_LEN], line[SRC_PORT], label,
                                                     browsing_type[label], directory[9:], in_out,
                                                     float(line[0]) - port_dt_dict_jap[line[SRC_PORT]]])
                        if directory == dirs[0]:
                            port_dt_dict_eng[line[SRC_PORT]] = float(line[0])
                        else:
                            port_dt_dict_jap[line[SRC_PORT]] = float(line[0])


def feature_aggregation(init_df):
    """

    :param init_df:
    :return:
    """
    stream_keys_dict = {}  # {key-port:value-stream_key}
    stream_key = []
    for index, row in init_df.iterrows():
        if row['tcp.srcport'] in stream_keys_dict.keys():
            stream_keys_dict[row['tcp.srcport']] += 1
        else:
            stream_keys_dict[row['tcp.srcport']] = 0
        stream_key.append(str(row['tcp.srcport']) + '-' +
                          str(int(stream_keys_dict[row['tcp.srcport']] / 10)))

    init_df = init_df.assign(stream_key=stream_key)
    init_df = count_delta_time_average_and_std_per_10_packets(init_df)
    init_df = average_and_std_packet_len_per_10_packets(init_df)

    return init_df


def count_delta_time_average_and_std_per_10_packets(init_df):
    average_values_dict = {}
    std_values_dict = {}

    for i_stream in sorted(init_df['stream_key'].unique()):
        mean = init_df.loc[init_df['stream_key'] == i_stream]['time_delta'].mean()
        std = init_df.loc[init_df['stream_key'] == i_stream]['time_delta'].std(ddof=0)
        average_values_dict[i_stream] = mean
        std_values_dict[i_stream] = std
    init_df['average_delta_time'] = init_df['stream_key'].apply(set_row_feature, args=(average_values_dict,))
    init_df['std_delta_time'] = init_df['stream_key'].apply(set_row_feature, args=(std_values_dict,))

    return init_df


def average_and_std_packet_len_per_10_packets(init_df):
    average_values_dict = {}
    std_values_dict = {}

    for i_stream in sorted(init_df['stream_key'].unique()):
        mean = init_df.loc[init_df['stream_key'] == i_stream]['frame.len'].mean()
        std = init_df.loc[init_df['stream_key'] == i_stream]['frame.len'].std(ddof=0)
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



def data_eng():
    # data_preprocessing()

    directory = '.\\place1_labeled'
    # directory = '.\\place2_labeled'
    files_dfs = []
    df = None
    for filename in os.listdir(directory):
        single_file_df = pd.read_csv(directory + '\\' + filename, sep='\t')
        # if single_file_df['categories'][0] == 'Browsing':
        #     df = feature_aggregation(single_file_df.head(60000))
        # elif single_file_df['categories'][0] == 'Chat':
        #     df = feature_aggregation(single_file_df.head(12000))
        # if single_file_df['categories'][0] == 'Streaming':
        #     df = feature_aggregation(single_file_df.head(20000))
        if single_file_df['categories'][0] == 'File Transfer':
            df = feature_aggregation(single_file_df.head(50000))
        # if single_file_df['categories'][0] == 'Video Conferencing':
        #     df = feature_aggregation(single_file_df.head(15000))

        files_dfs.append(df)
    final_labeled_df = pd.concat(files_dfs)
    final_labeled_df.to_csv('.\\FinalLabeledFileTransferEngland.tsv', index=False, sep='\t')
    # final_labeled_df.to_csv('.\\FinalLabeledFileTransferJapan.tsv', index=False, sep='\t')


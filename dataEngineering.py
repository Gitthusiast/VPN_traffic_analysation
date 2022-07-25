
import csv
import os
import pandas as pd

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


def data_preprocessing():

    # filename = '.\\place1\\England\\Dropbox_VPN_1.5hr_England.tsv'

        dirs = ['.\\place1\\England', '.\\place2\\Japan']
        for directory in dirs:

            # write all data into output labeled file
            with open(f'.\\labeled_{directory[9:]}.tsv', 'wt', newline='') as out_file:
                time = 0.0
                prev_time = 0.0
                tsv_writer = csv.writer(out_file, delimiter='\t')

                tsv_writer.writerow(['frame.time_epoch', 'frame.len', 'tcp.srcport',
                                     'Random website', 'Browsing', 'Country', 'time_delta'])

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

                            # count delta frame.time between consecutive packets
                            if i == 1:
                                tsv_writer.writerow([line[0], line[1], line[2], label, browsing_type[label],
                                                     directory[9:], 0])
                            else:
                                tsv_writer.writerow([line[0], line[1], line[2], label, browsing_type[label],
                                                     directory[9:], time-prev_time])


def data_preprocess(init_df):

    init_df.drop(columns=['ip.tos', 'tcp.options.mss_val', 'ip.opt.mtu'], inplace=True)
    # init_df['tcp.options.mss_val'].fillna(method='ffill', inplace=True)
    init_df.dropna(inplace=True)

    """init_df['tcp.srcport'] = np.where(init_df['tcp.srcport'] > 1024, 1, 0)
    init_df['tcp.dstport'] = np.where(init_df['tcp.dstport'] > 1024, 1, 0)"""

    init_df['ip.dsfield'] = init_df['ip.dsfield'].apply(str).apply(int, base=16)
    init_df['tcp.flags'] = init_df['tcp.flags'].apply(str).apply(int, base=16)
    init_df['ip.flags'] = init_df['ip.flags'].apply(str).apply(int, base=16)

    init_df.sort_values(['tcp.stream', 'frame.time_relative'], inplace=True, ignore_index=True)

    init_df = count_delta_time_average_and_std_per_10_packets(init_df)
    init_df = average_and_std_packet_len_per_10_packets(init_df)
    init_df.dropna(inplace=True)

    return init_df


def count_delta_time_average_and_std_per_10_packets(init_df):

    init_df['stream_key'] = 0
    stream_num = init_df['tcp.stream'][0]
    counter, stream_index = 1, 1

    for row in range(init_df.shape[0]):

        if init_df['tcp.stream'][row] != stream_num:
            stream_num = init_df['tcp.stream'][row]
            counter, stream_index = 1, stream_index + 1

        elif counter > 10:
            counter, stream_index = 1, stream_index + 1

        init_df.at[row, 'stream_key'] = stream_index
        counter += 1

    average_values_dict = {}
    std_values_dict = {}

    for i_stream in sorted(init_df['stream_key'].unique()):
        mean = init_df.loc[init_df['stream_key'] == i_stream]['tcp.time_delta'].mean()
        std = init_df.loc[init_df['stream_key'] == i_stream]['tcp.time_delta'].std()
        average_values_dict[i_stream] = mean
        std_values_dict[i_stream] = std
    init_df['average_time_delta'] = init_df['stream_key'].apply(set_row_feature, args=(average_values_dict,))
    init_df['std_time_delta'] = init_df['stream_key'].apply(set_row_feature, args=(std_values_dict,))

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
    # df = pd.read_csv('.\\labeled.tsv', sep='\t')


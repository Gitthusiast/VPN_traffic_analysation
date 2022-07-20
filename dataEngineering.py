
import csv
import os

browsing_type = {
    'Skype_chat': 'Chat',
    'Facebook': 'Chat',
    'GoogleHangouts': 'Chat',
    'Whatsapp_chat': 'Chat',
    'Telegram_chat': 'Chat',

    'Youtube': 'Streaming',
    'Netflix': 'Streaming',
    'Vimeo': 'Streaming',
    'qBittorrent': 'Streaming',
    'qBitTorrent': 'Streaming',

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
    # write all data into output labeled file
    with open('.\\labeled.tsv', 'wt', newline='') as out_file:
        time = 0.0
        prev_time = 0.0
        tsv_writer = csv.writer(out_file, delimiter='\t')

        tsv_writer.writerow(['frame.time_epoch', 'frame.len', 'tcp.srcport',
                             'Random website', 'Browsing', 'Country', 'time_delta'])

        dirs = ['.\\place1\\England', '.\\place2\\Japan']
        for directory in dirs:
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


if __name__ == '__main__':

    data_preprocessing()

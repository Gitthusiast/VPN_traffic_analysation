# VPN_traffic_analyzation
Methods for detecting cyber attacks

In the project we test our ability to classify the type
of the application/website visited by the user from his 
VPN encrypted traffic. We maintained a classification 
for both app categories and subcategories. 

###The categories:

    Chat
    Browsing
    Streaming
    File Transfer
    Video Conferencing

### The subcategories:

    Random_websites: Browsing

    Skype_chat: Chat
    Facebook: Chat
    GoogleHangouts: Chat
    Whatsapp_chat: Chat
    Telegram_chat: Chat

    Youtube: Streaming
    Netflix: Streaming
    Vimeo: Streaming

    qBittorrent: File Transfer
    Skype_files: File Transfer
    Dropbox: File Transfer
    gdrive: File Transfer
    Whatsapp_files: File Transfer
    Telegram_files: File Transfer

    Skype_video: Video Conferencing
    GoogleMeets: Video Conferencing
    Zoom: Video Conferencing
    MicrosoftTeams: Video Conferencing

###This project contains four main parts for VPN traffic classification
1. Data Engineering - which prepares the aggregated fields and labels the data. 
2. Feature Importance - to calculate the importance of all the features in the 
classification process
3. Feature Visualization - for better grasp on the feature distribution
4. Machine Learning - models training and evaluation
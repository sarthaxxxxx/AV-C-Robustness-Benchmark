echo "Begin audio extraction"
python extract_audio.py 
echo "Done with audio extraction"


sleep 3
echo "Begin frame extraction"
python extract_video_frame.py
echo "Done with frame extraction"
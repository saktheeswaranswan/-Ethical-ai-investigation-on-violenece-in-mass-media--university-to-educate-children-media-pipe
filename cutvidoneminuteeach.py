import os
from moviepy.editor import VideoFileClip

# Input folder containing all your .webm videos
input_folder = "/home/sakthees-monk/Music/karatatedrills"    # change this to your folder path
output_base = "output_videos"  # where you want the splitted videos

os.makedirs(output_base, exist_ok=True)

# Loop over all .webm files in the folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(".webm"):
        input_path = os.path.join(input_folder, file_name)
        # Load the video
        clip = VideoFileClip(input_path)
        duration = clip.duration  # total seconds

        # Make a folder for each video
        video_name = os.path.splitext(file_name)[0]
        video_output_folder = os.path.join(output_base, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Split into 1-minute (60 seconds) chunks
        chunk_length = 60  # seconds
        start = 0
        chunk_num = 1

        while start < duration:
            end = min(start + chunk_length, duration)
            subclip = clip.subclip(start, end)

            # Output file path (converted to .mp4)
            out_name = f"{video_name}_part{chunk_num}.mp4"
            out_path = os.path.join(video_output_folder, out_name)

            # Write as mp4
            subclip.write_videofile(out_path, codec="libx264", audio_codec="aac")
            chunk_num += 1
            start += chunk_length

        clip.close()

print("✅ All videos processed successfully!")


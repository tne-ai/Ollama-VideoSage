#!/usr/bin/env python
import cv2
import os
import numpy as np
import ollama
from datetime import datetime
import shutil
from tqdm import tqdm
import json
import asyncio
import edge_tts
import tempfile
import sys

class VideoAnalyzer:
    def __init__(self, config):
        """
        Initialize the video analyzer
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        # Video processing settings
        self.video_path = config['video_path']
        self.fps_sample_rate = config['fps_sample_rate']
        self.batch_size = config['batch_size']
        self.blur_threshold = config['blur_threshold']
        
        # AI Model settings
        self.vision_model = config['vision_model']
        self.text_model = config['text_model']
        
        # Prompts
        self.frame_prompt = config['prompts']['frame']
        self.segment_prompt = config['prompts']['segment']
        self.final_prompt = config['prompts']['final']
        
        # Create output directories
        self.base_dir = os.path.join(os.path.dirname(self.video_path), 'analysis_output')
        self.images_dir = os.path.join(self.base_dir, 'images')
        self.blurry_dir = os.path.join(self.base_dir, 'blurry')
        self.setup_directories()

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Results storage
        self.frame_descriptions = []
        self.segment_descriptions = []
        self.final_description = ""

    def setup_directories(self):
        """Create necessary directories for output files"""
        for dir_path in [self.base_dir, self.images_dir, self.blurry_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

    def detect_blur(self, image):
        """
        Detect if an image is blurry using Laplacian variance
        Returns: True if image is blurry, False otherwise
        """
        if image.shape[0] < 100 or image.shape[1] < 100:  # Skip very small images
            return True

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Print variance for debugging
        print(f"Image variance: {variance}")
        
        return variance < self.blur_threshold

    def extract_frames(self):
        """Extract frames from video and separate blurry from non-blurry frames"""
        frames_to_process = []
        frame_number = 0
        frames_per_sample = int(self.fps * self.fps_sample_rate)

        print("Extracting and analyzing frames...")
        with tqdm(total=self.frame_count) as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                pbar.update(1)
                
                # Process only frames at the specified sample rate
                if frame_number % frames_per_sample == 0:
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frame_path = os.path.join(
                        self.images_dir if not self.detect_blur(frame) else self.blurry_dir,
                        f'frame_{frame_number:06d}_{timestamp:.2f}s.jpg'
                    )
                    
                    cv2.imwrite(frame_path, frame)
                    if not self.detect_blur(frame):
                        frames_to_process.append((frame_number, frame_path, timestamp))

                frame_number += 1

        self.cap.release()
        return frames_to_process

    async def analyze_frame(self, frame_info):
        """Analyze a single frame using vision model"""
        frame_number, frame_path, timestamp = frame_info
        try:
            print(f"\nAnalyzing frame at {timestamp:.2f}s...")
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": self.frame_prompt,
                    "images": [frame_path]
                }]
            )
            
            description = response['message']['content']
            print(f"✓ Frame {frame_number} ({timestamp:.2f}s): {description[:100]}...")
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'description': description
            }
        except Exception as e:
            print(f"✗ Error analyzing frame {frame_number}: {e}")
            return None

    async def analyze_frames(self, frames_to_process):
        """Analyze all selected frames"""
        print(f"\nAnalyzing frames with {self.vision_model}...")
        self.frame_descriptions = []
        total_frames = len(frames_to_process)
        
        with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
            for i, frame_info in enumerate(frames_to_process, 1):
                frame_number = frame_info[0]
                pbar.set_description(f"Frame {i}/{total_frames} ({frame_number})")
                result = await self.analyze_frame(frame_info)
                if result:
                    self.frame_descriptions.append(result)
                pbar.update(1)
                pbar.set_postfix_str(f"Processed: {i}/{total_frames}")

    async def combine_descriptions(self, descriptions, prompt):
        """Combine descriptions using text model"""
        try:
            # Format differently based on whether we're handling frames or segments
            if 'frame_number' in descriptions[0]:  # Frame descriptions
                formatted_input = "\n".join([
                    f"Timestamp {d['timestamp']:.2f}s: {d['description']}"
                    for d in descriptions
                ])
            else:  # Segment descriptions
                formatted_input = "\n".join([
                    f"Segment {i+1} ({d['start_time']:.2f}s - {d['end_time']:.2f}s): {d['description']}"
                    for i, d in enumerate(descriptions)
                ])
            
            print("\nCombining descriptions...")
            response = ollama.chat(
                model=self.text_model,
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n\nDescriptions to combine:\n{formatted_input}"
                }]
            )
            combined = response['message']['content']
            print(f"✓ Combined description: {combined[:150]}...")
            return combined
        except Exception as e:
            print(f"✗ Error combining descriptions: {str(e)}")
            return None

    async def create_segment_descriptions(self):
        """Combine frame descriptions into segment descriptions"""
        print("\nCreating segment descriptions...")
        self.segment_descriptions = []
        total_segments = (len(self.frame_descriptions) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_segments, desc="Creating segments") as pbar:
            for i in range(0, len(self.frame_descriptions), self.batch_size):
                batch = self.frame_descriptions[i:i + self.batch_size]
                start_time = batch[0]['timestamp']
                end_time = batch[-1]['timestamp']
                pbar.set_description(f"Segment {i//self.batch_size + 1}/{total_segments} ({start_time:.1f}s - {end_time:.1f}s)")
                
                print(f"\nProcessing segment {i//self.batch_size + 1}/{total_segments} ({start_time:.1f}s - {end_time:.1f}s)")
                segment_desc = await self.combine_descriptions(
                    batch,
                    self.segment_prompt
                )
                if segment_desc:
                    self.segment_descriptions.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'description': segment_desc
                    })
                    print(f"✓ Segment {i//self.batch_size + 1} complete")
                else:
                    print(f"✗ Failed to create description for segment {i//self.batch_size + 1}")
                pbar.update(1)
                pbar.set_postfix_str(f"Completed: {len(self.segment_descriptions)}/{total_segments}")

    async def create_final_description(self):
        """Create final video description from segment descriptions"""
        print("\nCreating final video description...")
        if not self.segment_descriptions:
            self.final_description = "Failed to generate video description."
            print("✗ No segment descriptions available to create final description")
            return

        print(f"Combining {len(self.segment_descriptions)} segment descriptions into final summary...")
        final_desc = await self.combine_descriptions(
            self.segment_descriptions,
            self.final_prompt
        )
        
        if final_desc:
            self.final_description = final_desc
            print("\n✓ Final description created successfully:")
            print("-" * 80)
            print(self.final_description)
            print("-" * 80)
        else:
            self.final_description = "Failed to generate video description due to an error."
            print("✗ Failed to create final description")

    def save_results(self):
        """Save analysis results to a JSON file"""
        results = {
            'video_path': self.video_path,
            'analysis_parameters': {
                'fps_sample_rate': self.fps_sample_rate,
                'batch_size': self.batch_size,
                'blur_threshold': self.blur_threshold
            },
            'frame_descriptions': self.frame_descriptions,
            'segment_descriptions': self.segment_descriptions,
            'final_description': self.final_description
        }
        
        output_path = os.path.join(self.base_dir, 'analysis_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    async def speak_description(self, text):
        """Speak the given text using edge-tts"""
        try:
            print("\nSpeaking final description...")
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, "final_description.mp3")
            
            communicate = edge_tts.Communicate(text)
            await communicate.save(audio_path)
            
            # Use system audio player
            if os.name == 'posix':  # macOS or Linux
                process = await asyncio.create_subprocess_exec(
                    'afplay' if os.uname().sysname == 'Darwin' else 'aplay',
                    audio_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
            else:  # Windows
                os.system(f'start {audio_path}')
            
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            print("✓ Finished speaking description")
        except Exception as e:
            print(f"✗ Error speaking description: {str(e)}")

    async def analyze_video(self):
        """Main method to run the complete video analysis"""
        print(f"\nAnalyzing video: {self.video_path}")
        print(f"Video FPS: {self.fps}")
        print(f"Total frames: {self.frame_count}")
        print(f"Sampling 1 frame every {self.fps_sample_rate} second(s)")
        
        frames_to_process = self.extract_frames()
        await self.analyze_frames(frames_to_process)
        await self.create_segment_descriptions()
        await self.create_final_description()
        self.save_results()
        
        print("\nAnalysis complete!")
        print(f"Final Description:\n{self.final_description}")
        
        # Speak the final description
        await self.speak_description(self.final_description)

async def main():
    # Check if video file is provided as argument
    if len(sys.argv) != 2:
        print("Usage: python vision.py <video_file>")
        print("Example: python vision.py video.mp4")
        sys.exit(1)

    # Get video path from command line argument
    video_name = sys.argv[1]
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_name)

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_name}' not found in the current directory")
        sys.exit(1)
    
    # ==============================================
    # Configuration Section - Adjust settings here
    # ==============================================
    
    config = {
        # Video Processing Settings
        "video_path": video_path,
        "fps_sample_rate": 5,    # Analyze 1 frame every 5 seconds
        "batch_size": 8,         # Combine 8 frames per segment
        "blur_threshold": 4,     # Blur detection sensitivity
        
        # AI Models
        "vision_model": "minicpm-v:latest",  # Model for analyzing frames
        "text_model": "command-r7b:latest",  # Model for combining descriptions
        
        # Prompts
        "prompts": {
            # Prompt for analyzing individual frames
            "frame": "Describe what is happening in this frame of the video. Focus on the main action or subject.",
            
            # Prompt for combining frames into segment descriptions
            "segment": "Combine these frame descriptions into a coherent description of this video segment. "
                      "Focus on the main events and progression of action."
                      "NEVER mention any times or dates.",
            
            # Prompt for creating the final video description
            "final": "Create a concise summary of the entire video based on these segment descriptions. "
                    "Provide a coherent narrative that captures the main events and progression of the video. "
                    "NEVER mention any times or dates."
                    "This is a single scene video, so don't mention anything about other scenes."
                    "Do not include any additional information, just the summary."
        }
    }
    
    # Initialize and run the analyzer
    analyzer = VideoAnalyzer(config)
    await analyzer.analyze_video()

if __name__ == "__main__":
    asyncio.run(main()) 
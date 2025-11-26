"""
Music Player Module

This module handles playing emotion-specific music files.
"""

import os
import random
import threading
from pathlib import Path
from typing import Optional, List
import pygame


class EmotionMusicPlayer:
    """
    Music player that plays emotion-specific songs.
    
    This player manages a library of music organized by emotion
    and plays appropriate songs based on the detected emotion.
    """
    
    def __init__(self, music_dir: str = 'music', volume: float = 0.5):
        """
        Initialize the music player.
        
        Args:
            music_dir: Directory containing emotion-specific music folders
            volume: Initial volume (0.0 to 1.0)
        """
        self.music_dir = Path(music_dir)
        self.current_emotion = None
        self.current_song = None
        self.volume = volume
        self.is_playing = False
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            pygame.mixer.music.set_volume(volume)
            self._initialized = True
            print(f"Music player initialized (volume: {volume:.0%})")
        except Exception as e:
            print(f"Warning: Could not initialize music player: {e}")
            self._initialized = False
        
        # Scan music library
        self.music_library = self._scan_music_library()
        self._print_library_info()
    
    def _scan_music_library(self) -> dict:
        """
        Scan the music directory for emotion-specific songs.
        
        Returns:
            Dictionary mapping emotions to lists of song paths
        """
        library = {
            'angry': [],
            'sad': [],
            'happy': [],
            'neutral': []
        }
        
        if not self.music_dir.exists():
            print(f"Warning: Music directory not found: {self.music_dir}")
            print("Creating empty music directories...")
            for emotion in library.keys():
                emotion_dir = self.music_dir / emotion
                emotion_dir.mkdir(parents=True, exist_ok=True)
            return library
        
        # Supported audio formats
        audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
        
        # Scan each emotion folder
        for emotion in library.keys():
            emotion_dir = self.music_dir / emotion
            
            if emotion_dir.exists() and emotion_dir.is_dir():
                # Find all audio files
                for file_path in emotion_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                        library[emotion].append(str(file_path))
        
        return library
    
    def _print_library_info(self):
        """Print information about the music library."""
        print("\nMusic Library:")
        total_songs = 0
        for emotion, songs in self.music_library.items():
            count = len(songs)
            total_songs += count
            status = f"{count} song(s)" if count > 0 else "empty"
            print(f"  {emotion:8s}: {status}")
        
        if total_songs == 0:
            print("\nNote: No music files found. Place audio files in:")
            for emotion in self.music_library.keys():
                print(f"  - music/{emotion}/")
        print()
    
    def play_emotion(self, emotion: str, shuffle: bool = True):
        """
        Play a song matching the given emotion.
        
        Args:
            emotion: Emotion label ('angry', 'sad', 'happy', 'neutral')
            shuffle: Whether to pick a random song from the emotion folder
        """
        if not self._initialized:
            return
        
        # Validate emotion
        if emotion not in self.music_library:
            print(f"Warning: Unknown emotion: {emotion}")
            return
        
        # Check if we're already playing the right emotion
        if self.current_emotion == emotion and self.is_playing:
            # Check if song is still playing
            if pygame.mixer.music.get_busy():
                return  # Continue playing current song
        
        # Get songs for this emotion
        songs = self.music_library[emotion]
        
        if len(songs) == 0:
            print(f"No music files found for emotion: {emotion}")
            return
        
        # Select song
        if shuffle:
            song_path = random.choice(songs)
        else:
            song_path = songs[0]
        
        # Play song
        try:
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play(-1)  # Loop indefinitely
            
            self.current_emotion = emotion
            self.current_song = Path(song_path).name
            self.is_playing = True
            
            print(f"♪ Now playing ({emotion}): {self.current_song}")
        
        except Exception as e:
            print(f"Error playing song: {e}")
    
    def stop(self):
        """Stop the currently playing music."""
        if not self._initialized:
            return
        
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.current_emotion = None
            self.current_song = None
            print("Music stopped")
    
    def pause(self):
        """Pause the currently playing music."""
        if not self._initialized:
            return
        
        if self.is_playing:
            pygame.mixer.music.pause()
            print("Music paused")
    
    def resume(self):
        """Resume paused music."""
        if not self._initialized:
            return
        
        pygame.mixer.music.unpause()
        print("Music resumed")
    
    def set_volume(self, volume: float):
        """
        Set the music volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self._initialized:
            return
        
        self.volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.volume)
    
    def get_status(self) -> dict:
        """
        Get current player status.
        
        Returns:
            Dictionary with player status information
        """
        return {
            'is_playing': self.is_playing,
            'current_emotion': self.current_emotion,
            'current_song': self.current_song,
            'volume': self.volume,
            'library_size': {
                emotion: len(songs)
                for emotion, songs in self.music_library.items()
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self._initialized:
            self.stop()
            pygame.mixer.quit()


if __name__ == "__main__":
    """
    Test the music player.
    """
    import time
    
    print("=" * 80)
    print("Music Player Test")
    print("=" * 80)
    
    # Initialize player
    player = EmotionMusicPlayer(music_dir='music', volume=0.3)
    
    # Test playing different emotions
    emotions = ['happy', 'sad', 'angry', 'neutral']
    
    print("\nTesting emotion switching...")
    print("(Note: You need to have music files in the music/ folders)")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        for i, emotion in enumerate(emotions * 2):  # Test each emotion twice
            print(f"\n[Test {i+1}] Switching to emotion: {emotion}")
            player.play_emotion(emotion)
            
            # Display status
            status = player.get_status()
            print(f"Status: Playing={status['is_playing']}, "
                  f"Emotion={status['current_emotion']}, "
                  f"Song={status['current_song']}")
            
            # Wait a bit
            time.sleep(3)
        
        print("\nTest sequence completed!")
        time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    finally:
        print("\nCleaning up...")
        player.cleanup()
        print("Music player test completed!")

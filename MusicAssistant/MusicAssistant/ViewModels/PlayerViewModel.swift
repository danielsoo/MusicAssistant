//
//  PlayerViewModel.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import Foundation
import Observation

@Observable
class PlayerViewModel {
    var currentSong: Song?
    var isPlaying: Bool = false
    var currentTime: TimeInterval = 0
    var queue: [Song] = []
    var currentQueue: [Song] = []
    var currentIndex: Int = 0
    
    var progress: Double {
        guard let song = currentSong, song.duration > 0 else { return 0 }
        return currentTime / song.duration
    }
    
    func playSong(_ song: Song, from songs: [Song] = []) {
        currentSong = song
        isPlaying = true
        currentTime = 0
        
        if !songs.isEmpty {
            currentQueue = songs
            currentIndex = songs.firstIndex(where: { $0.id == song.id }) ?? 0
        }
    }
    
    func togglePlayPause() {
        isPlaying.toggle()
    }
    
    func playNext() {
        guard !currentQueue.isEmpty else { return }
        currentIndex = (currentIndex + 1) % currentQueue.count
        currentSong = currentQueue[currentIndex]
        isPlaying = true
        currentTime = 0
    }
    
    func playPrevious() {
        guard !currentQueue.isEmpty else { return }
        currentIndex = (currentIndex - 1 + currentQueue.count) % currentQueue.count
        currentSong = currentQueue[currentIndex]
        isPlaying = true
        currentTime = 0
    }
    
    func seek(to time: TimeInterval) {
        currentTime = time
    }
}

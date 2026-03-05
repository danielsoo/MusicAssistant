//
//  Models.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import Foundation

struct Song: Identifiable, Hashable {
    let id = UUID()
    let title: String
    let artist: String
    let albumArt: String
    let duration: TimeInterval
    let albumName: String
    
    var durationFormatted: String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

struct Playlist: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let coverImage: String
    let songs: [Song]
    let creator: String
    
    var songCount: String {
        "\(songs.count) songs"
    }
}

struct Album: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let artist: String
    let coverImage: String
    let songs: [Song]
    let year: Int
}

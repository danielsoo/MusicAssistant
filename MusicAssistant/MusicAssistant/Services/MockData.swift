//
//  MockData.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import Foundation

struct MockData {
    static let songs: [Song] = [
        Song(title: "Blinding Lights", artist: "The Weeknd", albumArt: "album1", duration: 200, albumName: "After Hours"),
        Song(title: "Levitating", artist: "Dua Lipa", albumArt: "album2", duration: 203, albumName: "Future Nostalgia"),
        Song(title: "Save Your Tears", artist: "The Weeknd", albumArt: "album1", duration: 215, albumName: "After Hours"),
        Song(title: "Good 4 U", artist: "Olivia Rodrigo", albumArt: "album3", duration: 178, albumName: "SOUR"),
        Song(title: "Stay", artist: "The Kid LAROI", albumArt: "album4", duration: 141, albumName: "F*ck Love 3"),
        Song(title: "Peaches", artist: "Justin Bieber", albumArt: "album5", duration: 198, albumName: "Justice"),
        Song(title: "Kiss Me More", artist: "Doja Cat", albumArt: "album6", duration: 208, albumName: "Planet Her"),
        Song(title: "Montero", artist: "Lil Nas X", albumArt: "album7", duration: 137, albumName: "Montero"),
    ]
    static let playlists: [Playlist] = [
        Playlist(name: "Today's Top Hits", coverImage: "playlist1", songs: Array(songs.prefix(5)), creator: "Spotify"),
        Playlist(name: "RapCaviar", coverImage: "playlist2", songs: Array(songs.suffix(4)), creator: "Spotify"),
        Playlist(name: "Rock Classics", coverImage: "playlist3", songs: Array(songs.prefix(3)), creator: "Spotify"),
        Playlist(name: "Chill Vibes", coverImage: "playlist4", songs: songs, creator: "Spotify"),
    ]
    
    static let albums: [Album] = [
        Album(name: "After Hours", artist: "The Weeknd", coverImage: "album1", songs: [songs[0], songs[2]], year: 2020),
        Album(name: "Future Nostalgia", artist: "Dua Lipa", coverImage: "album2", songs: [songs[1]], year: 2020),
        Album(name: "SOUR", artist: "Olivia Rodrigo", coverImage: "album3", songs: [songs[3]], year: 2021),
    ]
    
    static let recentlyPlayed: [Song] = Array(songs.prefix(6))
}



//
//  ReusableComponents.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct SongRowView: View {
    let song: Song
    var showAlbumArt: Bool = true
    
    var body: some View {
        HStack(spacing: 12) {
            if showAlbumArt {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.gray)
                    .frame(width: 50, height: 50)
                    .overlay {
                        Image(systemName: "music.note")
                            .foregroundStyle(.white.opacity(0.6))
                    }
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(song.title)
                    .foregroundStyle(.white)
                    .lineLimit(1)
                
                Text(song.artist)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            
            Spacer()
            
            Text(song.durationFormatted)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

struct PlaylistSection: View {
    let title: String
    let playlists: [Playlist]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title2)
                .bold()
                .padding(.horizontal)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(playlists) { playlist in
                        NavigationLink(value: playlist) {
                            PlaylistCard(playlist: playlist)
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct PlaylistCard: View {
    let playlist: Playlist
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.gradient)
                .frame(width: 150, height: 150)
                .overlay {
                    Image(systemName: "music.note.list")
                        .font(.system(size: 50))
                        .foregroundStyle(.white.opacity(0.6))
                }
            
            Text(playlist.name)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .lineLimit(2)
                .frame(width: 150, alignment: .leading)
        }
    }
}

struct GenreTile: View {
    let playlist: Playlist
    
    var body: some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(Color.random.gradient)
            .frame(height: 100)
            .overlay(alignment: .topLeading) {
                Text(playlist.name)
                    .font(.headline)
                    .foregroundStyle(.white)
                    .padding()
            }
    }
}

struct RecentlyPlayedSection: View {
    
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
            ForEach(Array(MockData.recentlyPlayed.prefix(4))) { song in
                HStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray)
                        .frame(width: 60, height: 60)
                        .overlay {
                            Image(systemName: "music.note")
                                .foregroundStyle(.white.opacity(0.6))
                        }
                    
                    Text(song.title)
                        .font(.footnote)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                        .lineLimit(2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 60)
                .background(Color(white: 0.15))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .onTapGesture {
                    playerViewModel.playSong(song, from: MockData.recentlyPlayed)
                }
            }
        }
        .padding(.horizontal)
    }
}

struct LibraryPlaylistRow: View {
    let playlist: Playlist
    
    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.gray)
                .frame(width: 60, height: 60)
                .overlay {
                    Image(systemName: "music.note.list")
                        .foregroundStyle(.white.opacity(0.6))
                }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(playlist.name)
                    .foregroundStyle(.white)
                
                Text("Playlist • \(playlist.songCount)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
        }
        .padding(.horizontal)
    }
}

struct LibraryAlbumRow: View {
    let album: Album
    
    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.purple.gradient)
                .frame(width: 60, height: 60)
                .overlay {
                    Image(systemName: "opticaldisc")
                        .foregroundStyle(.white.opacity(0.6))
                }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(album.name)
                    .foregroundStyle(.white)
                
                Text("Album • \(album.artist)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
        }
        .padding(.horizontal)
    }
}

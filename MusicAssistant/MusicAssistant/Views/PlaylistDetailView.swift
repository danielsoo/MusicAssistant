//
//  PlaylistDetailView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct PlaylistDetailView: View {
    
    let playlist: Playlist
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                
                // 🎨 Cover
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.gradient)
                    .frame(height: 300)
                    .overlay {
                        Image(systemName: "music.note.list")
                            .font(.system(size: 80))
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .padding()
                
                // 📌 Metadata
                VStack(alignment: .leading, spacing: 8) {
                    Text(playlist.name)
                        .font(.title)
                        .bold()
                    
                    Text(playlist.creator)
                        .foregroundStyle(.secondary)
                    
                    Text(playlist.songCount)
                        .foregroundStyle(.secondary)
                        .font(.subheadline)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                
                // ▶️ Controls
                HStack(spacing: 24) {
                    
                    Button(action: {
                        playAll()
                    }) {
                        Image(systemName: "play.fill")
                            .font(.title2)
                            .foregroundStyle(.black)
                            .frame(width: 56, height: 56)
                            .background(Color.spotifyGreen)
                            .clipShape(Circle())
                    }
                    
                    Button(action: {
                        // Shuffle optional for now
                    }) {
                        Image(systemName: "shuffle")
                            .font(.title2)
                            .foregroundStyle(Color.spotifyGreen)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                
                // 🎵 Song List
                ForEach(playlist.songs) { song in
                    SongRowView(song: song, showAlbumArt: false)
                        .onTapGesture {
                            playerViewModel.playSong(song, from: playlist.songs)
                        }
                }
                .padding(.bottom, 100)
            }
        }
        .background(Color.black)
        .scrollIndicators(.hidden)
    }
    
    // ▶️ Play All
    private func playAll() {
        guard let firstSong = playlist.songs.first else { return }
        playerViewModel.playSong(firstSong, from: playlist.songs)
    }
}

//
//  AlbumDetailView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct AlbumDetailView: View {
    
    let album: Album
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                
                // 💿 Album Cover
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.purple.gradient)
                    .frame(height: 300)
                    .overlay {
                        Image(systemName: "opticaldisc")
                            .font(.system(size: 80))
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .padding()
                
                // 📌 Metadata
                VStack(alignment: .leading, spacing: 8) {
                    Text(album.name)
                        .font(.title)
                        .bold()
                    
                    Text(album.artist)
                        .foregroundStyle(.secondary)
                    
                    Text("Album • \(album.year)")
                        .foregroundStyle(.secondary)
                        .font(.subheadline)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                
                // ▶️ Play Button
                Button(action: {
                    playAll()
                }) {
                    Label("Play", systemImage: "play.fill")
                        .font(.headline)
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(Color.spotifyGreen)
                        .clipShape(RoundedRectangle(cornerRadius: 25))
                }
                .padding()
                
                // 🎵 Song List
                ForEach(album.songs) { song in
                    SongRowView(song: song, showAlbumArt: false)
                        .onTapGesture {
                            playerViewModel.playSong(song, from: album.songs)
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
        guard let firstSong = album.songs.first else { return }
        playerViewModel.playSong(firstSong, from: album.songs)
    }
}

#Preview {
    NavigationStack {
        AlbumDetailView(album: MockData.albums[0])
            .environment(PlayerViewModel())
    }
}

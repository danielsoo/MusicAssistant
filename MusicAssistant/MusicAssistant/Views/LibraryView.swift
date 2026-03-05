//
//  LibraryView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct LibraryView: View {
    
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    
                    // 🎵 Playlists Section
                    Text("Playlists")
                        .font(.title2)
                        .bold()
                        .padding(.horizontal)
                    
                    ForEach(MockData.playlists) { playlist in
                        NavigationLink(value: playlist) {
                            LibraryPlaylistRow(playlist: playlist)
                        }
                    }
                    
                    // 💿 Albums Section
                    Text("Albums")
                        .font(.title2)
                        .bold()
                        .padding(.horizontal)
                        .padding(.top)
                    
                    ForEach(MockData.albums) { album in
                        NavigationLink(value: album) {
                            LibraryAlbumRow(album: album)
                        }
                    }
                }
                .padding(.bottom, 100)
            }
            .background(Color.black)
            .navigationTitle("Your Library")
            .scrollIndicators(.hidden)
            
            // 🔗 Navigation destinations
            .navigationDestination(for: Playlist.self) { playlist in
                PlaylistDetailView(playlist: playlist)
            }
            
            .navigationDestination(for: Album.self) { album in
                Text(album.name)
                    .foregroundStyle(.white)
            }
        }
    }
}

#Preview {
    LibraryView()
        .environment(PlayerViewModel())
}

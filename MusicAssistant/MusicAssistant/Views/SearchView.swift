//
//  SearchView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct SearchView: View {
    
    @State private var searchText = ""
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    // 필터링된 곡 리스트
    var filteredSongs: [Song] {
        if searchText.isEmpty {
            return MockData.songs
        }
        
        return MockData.songs.filter {
            $0.title.localizedCaseInsensitiveContains(searchText) ||
            $0.artist.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                
                if searchText.isEmpty {
                    
                    // 🔎 Browse Categories
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            
                            Text("Browse All")
                                .font(.title2)
                                .bold()
                                .padding(.horizontal)
                            
                            LazyVGrid(
                                columns: [GridItem(.flexible()), GridItem(.flexible())],
                                spacing: 16
                            ) {
                                ForEach(MockData.playlists) { playlist in
                                    NavigationLink(value: playlist) {
                                        GenreTile(playlist: playlist)
                                    }
                                }
                            }
                            .padding(.horizontal)
                        }
                        .padding(.bottom, 100)
                    }
                    
                } else {
                    
                    // 🎵 Search Results
                    List(filteredSongs) { song in
                        SongRowView(song: song)
                            .listRowBackground(Color.black)
                            .onTapGesture {
                                playerViewModel.playSong(song, from: filteredSongs)
                            }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .searchable(text: $searchText, prompt: "Songs, artists, or playlists")
            .navigationTitle("Search")
            .background(Color.black)
            
            // 🔗 Navigation Destination
            .navigationDestination(for: Playlist.self) { playlist in
                Text(playlist.name)
                    .foregroundStyle(.white)
            }
        }
    }
}

#Preview {
    SearchView()
        .environment(PlayerViewModel())
}

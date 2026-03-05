//
//  HomeView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct HomeView: View {
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    
                    // 🔥 인사말
                    Text("Good \(timeOfDay())")
                        .font(.largeTitle)
                        .bold()
                        .padding(.horizontal)
                    
                    // 🔥 최근 재생
                    RecentlyPlayedSection()
                    
                    // 🔥 Made For You
                    PlaylistSection(
                        title: "Made For You",
                        playlists: Array(MockData.playlists.prefix(2))
                    )
                    
                    // 🔥 Popular
                    PlaylistSection(
                        title: "Popular Playlists",
                        playlists: Array(MockData.playlists.suffix(2))
                    )
                }
                .padding(.bottom, 100)
            }
            .background(Color.black)
            .scrollIndicators(.hidden)
            
            // ✅ 이거 추가!!!
            .navigationDestination(for: Playlist.self) { playlist in
                PlaylistDetailView(playlist: playlist)
            }
        }
    }
}

private func timeOfDay() -> String {
    let hour = Calendar.current.component(.hour, from: Date())
    
    switch hour {
    case 0..<12:
        return "morning"
    case 12..<17:
        return "afternoon"
    default:
        return "evening"
    }
}

#Preview {
    HomeView()
        .environment(PlayerViewModel())
}

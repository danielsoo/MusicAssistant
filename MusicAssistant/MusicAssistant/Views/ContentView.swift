//
//  ContentView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        ZStack(alignment: .bottom) {
            
            TabView(selection: $selectedTab) {
                
                HomeView()
                    .tabItem {
                        Label("Home", systemImage: "house.fill")
                    }
                    .tag(0)
                
                SearchView()
                    .tabItem {
                        Label("Search", systemImage: "magnifyingglass")
                    }
                    .tag(1)
                
                LibraryView()
                    .tabItem {
                        Label("Library", systemImage: "books.vertical.fill")
                    }
                    .tag(2)
            }
            .tint(.spotifyGreen)
            
            // 🔥 조건부 Mini Player
            if playerViewModel.currentSong != nil {
                MiniPlayerView()
                    .padding(.bottom, 49)
            }
        }
    }
}
#Preview {
    ContentView()
        .environment(PlayerViewModel())
}

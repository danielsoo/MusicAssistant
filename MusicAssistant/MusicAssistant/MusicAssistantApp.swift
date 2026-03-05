//
//  MusicAssistantApp.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

@main
struct MusicAssistantApp: App {
    @State private var playerViewModel = PlayerViewModel()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(playerViewModel)
        }
    }
}

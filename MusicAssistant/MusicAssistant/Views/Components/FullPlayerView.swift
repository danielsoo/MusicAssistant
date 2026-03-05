//
//  FullPlayerView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct FullPlayerView: View {
    
    @Environment(\.dismiss) private var dismiss
    @Environment(PlayerViewModel.self) private var playerViewModel
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 24) {
                
                HStack {
                    Button(action: { dismiss() }) {
                        Image(systemName: "chevron.down")
                            .font(.title2)
                            .foregroundStyle(.white)
                    }
                    Spacer()
                }
                .padding(.horizontal)
                
                Spacer()
                
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.gray.gradient)
                    .frame(width: 320, height: 320)
                    .overlay {
                        Image(systemName: "music.note")
                            .font(.system(size: 100))
                            .foregroundStyle(.white.opacity(0.6))
                    }
                    .shadow(radius: 20)
                
                VStack(spacing: 8) {
                    Text(playerViewModel.currentSong?.title ?? "")
                        .font(.title2)
                        .bold()
                        .foregroundStyle(.white)
                    
                    Text(playerViewModel.currentSong?.artist ?? "")
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                
                Spacer()
            }
        }
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview {
    FullPlayerView()
        .environment(PlayerViewModel())
}

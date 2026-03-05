//
//  MiniPlayerView.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

struct MiniPlayerView: View {
    @Environment(PlayerViewModel.self) private var playerViewModel
    @State private var showFullPlayer = false
    
    var body: some View {
        Button(action: {
            showFullPlayer = true
        }) {
            HStack(spacing: 12) {
                
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.gray)
                    .frame(width: 50, height: 50)
                    .overlay {
                        Image(systemName: "music.note")
                            .foregroundStyle(.white.opacity(0.6))
                    }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(playerViewModel.currentSong?.title ?? "")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                        .lineLimit(1)
                    
                    Text(playerViewModel.currentSong?.artist ?? "")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                
                Spacer()
                
                Button(action: {
                    playerViewModel.togglePlayPause()
                }) {
                    Image(systemName: playerViewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                        .foregroundStyle(.white)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(white: 0.15))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 8)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showFullPlayer) {
            FullPlayerView()
        }
    }
}

#Preview {
    MiniPlayerView()
        .environment(PlayerViewModel())
        .background(Color.black)
}

//
//  ColorExtensions.swift
//  MusicAssistant
//
//  Created by YounSoo Park on 01/03/2026.
//

import SwiftUI

extension Color {
    static let spotifyGreen = Color(red: 0.114, green: 0.835, blue: 0.329)
    
    static var random: Color {
        let colors: [Color] = [
            .blue, .purple, .pink, .red, .orange, .green, .indigo, .cyan
        ]
        return colors.randomElement() ?? .blue
    }
}

import React, { useMemo, useState } from 'react';
import { ActivityIndicator, Platform, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { WebView } from 'react-native-webview';
import type { WebViewErrorEvent } from 'react-native-webview/lib/WebViewTypes';

function getDefaultUrl(): string {
  if (Platform.OS === 'android') return 'http://10.0.2.2:8000';
  return 'http://localhost:8000';
}

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const appUrl = useMemo(() => {
    const envUrl = (globalThis as any)?.process?.env?.EXPO_PUBLIC_APP_URL as string | undefined;
    return envUrl ?? getDefaultUrl();
  }, []);

  function handleWebError(e: WebViewErrorEvent) {
    setError(e.nativeEvent.description || 'Network error');
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.title}>Music Assistant App</Text>
        <Text style={styles.subtitle}>{appUrl}</Text>
      </View>

      {error ? (
        <View style={styles.errorWrap}>
          <Text style={styles.errorTitle}>Cannot connect to server</Text>
          <Text style={styles.errorText}>{error}</Text>
          <Text style={styles.errorText}>Set EXPO_PUBLIC_APP_URL to your server URL.</Text>
        </View>
      ) : (
        <WebView
          source={{ uri: appUrl }}
          onError={handleWebError}
          startInLoadingState
          renderLoading={() => (
            <View style={styles.loadingWrap}>
              <ActivityIndicator size="large" />
              <Text style={styles.loadingText}>Loading site...</Text>
            </View>
          )}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  header: {
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#fafafa',
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  subtitle: {
    marginTop: 2,
    fontSize: 12,
    color: '#6b7280',
  },
  loadingWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  loadingText: {
    color: '#6b7280',
  },
  errorWrap: {
    flex: 1,
    padding: 16,
    justifyContent: 'center',
  },
  errorTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 10,
    color: '#111827',
  },
  errorText: {
    fontSize: 14,
    marginBottom: 6,
    color: '#374151',
  },
});

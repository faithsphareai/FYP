import { Stack } from "expo-router";
import { Suspense, useEffect } from 'react';
import * as SplashScreen from 'expo-splash-screen';
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StatusBar } from "expo-status-bar";
import { AuthProvider } from '../context/AuthContext';
import { SQLiteProvider, openDatabaseSync } from 'expo-sqlite';
import { drizzle } from 'drizzle-orm/expo-sqlite';
import { ActivityIndicator } from "react-native";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      cacheTime: 300000, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export const DATABASE_NAME = 'users';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const expoDb = openDatabaseSync(DATABASE_NAME);
  const db = drizzle(expoDb);

  useEffect(() => {
    const hideSplash = async () => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await SplashScreen.hideAsync();
    };

    hideSplash();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <SQLiteProvider
          databaseName={DATABASE_NAME}
          options={{ enableChangeListener: true }}
          useSuspense={false}>
              <GestureHandlerRootView style={{ flex: 1 }}>
                <StatusBar style="light" />
                <Stack>
                  <Stack.Screen name="index" options={{ headerShown: false }} />
                  <Stack.Screen name="(root)" options={{ headerShown: false }} />
                  <Stack.Screen name="(auth)" options={{ headerShown: false }} />
                </Stack>
              </GestureHandlerRootView>
        </SQLiteProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

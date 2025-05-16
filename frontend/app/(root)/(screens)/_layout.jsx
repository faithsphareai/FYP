import { Stack } from 'expo-router';

export default function Layout() {
  return (
    <Stack>
      <Stack.Screen
        name="mood-questionnaire"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="home"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="recitation"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="ocr"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="authentication-screen"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="namaz-tariqa"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="monthly-prayer-timings"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="get-nearby-mosques"
        options={{ headerShown: false }}
      />
      <Stack.Screen
        name="mosque-modal"
        options={{
          headerShown: false,
          presentation: 'transparentModal',
          gestureEnabled: true,
          animation: 'fade'
        }}
      />
      <Stack.Screen
        name="chatbot"
        options={({ route }) => ({
          headerShown: false,
          title: typeof route?.params?.title === 'string' ? route.params.title : 'Chat'
        })}
      />
    </Stack>
  );
}
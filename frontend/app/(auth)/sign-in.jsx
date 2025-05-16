import { 
  View, 
  Text, 
  Image, 
  TextInput, 
  TouchableOpacity, 
  ScrollView, 
  KeyboardAvoidingView, 
  Platform,
  Alert,
  ActivityIndicator
} from 'react-native';
import { useRouter } from 'expo-router';
import { images } from '../../constants'; 
import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
// Replace database import with authService
import authService from '../../utils/services/authService';

export default function LoginScreen() {
  const router = useRouter();
  const { login, isAuthenticated } = useAuth();
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Check if user is already logged in
  useEffect(() => {
    if (isAuthenticated()) {
      router.replace('(root)/(screens)/home');
    }
  }, [isAuthenticated, router]);

  const handleSignIn = async () => {
    try {
      // Validate inputs
      if (!email || !password) {
        Alert.alert('Error', 'Please enter both email and password');
        return;
      }
      
      setLoading(true);
      
      // Use the API service to authenticate
      const userData = await authService.login(email, password);
      
      setLoading(false);
      
      if (!userData) {
        Alert.alert('Error', 'Invalid email or password');
        return;
      }
      
      // Store user in AuthContext
      await login(userData);
      
      // Reset form
      setEmail('');
      setPassword('');
      
      // Navigate to home screen
      router.replace('(root)/(screens)/home');
      
    } catch (error) {
      setLoading(false);
      
      let errorMessage = 'Failed to login. Please try again.';
      
      if (error.response && error.response.data && error.response.data.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      Alert.alert('Error', errorMessage);
    }
  };

  return (
    <KeyboardAvoidingView 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={{ flex: 1, backgroundColor: 'white' }}
    >
      <ScrollView 
        contentContainerStyle={{ flexGrow: 1 }}
        showsVerticalScrollIndicator={false}
        style={{ paddingHorizontal: 24 }}
        keyboardShouldPersistTaps="handled"
      >
        {/* Login Image */}
        <View style={{ alignItems: 'center', marginTop: 80 }}>
          <Image 
            source={images.login} 
            resizeMode="contain"
            style={{ width: 350, height: 350 }}
          />
        </View>

        {/* Form Section - Vertically Centered */}
        <View style={{ flex: 1, justifyContent: 'center', gap: 24 }}>
          {/* Title */}
          <Text style={{ 
            fontSize: 24, 
            fontWeight: 'bold', 
            textAlign: 'center', 
            color: '#1f2937' 
          }}>
            Welcome Back
          </Text>

          {/* Email Field */}
          <View style={{ borderBottomWidth: 1, borderBottomColor: '#d1d5db', paddingBottom: 8 }}>
            <TextInput
              placeholder="Email"
              placeholderTextColor="#A0AEC0"
              keyboardType="email-address"
              autoCapitalize="none"
              style={{ fontSize: 18, color: '#4b5563' }}
              value={email}
              onChangeText={setEmail}
            />
          </View>

          {/* Password Field */}
          <View style={{ borderBottomWidth: 1, borderBottomColor: '#d1d5db', paddingBottom: 8 }}>
            <TextInput
              placeholder="Password"
              placeholderTextColor="#A0AEC0"
              secureTextEntry
              style={{ fontSize: 18, color: '#4b5563' }}
              value={password}
              onChangeText={setPassword}
            />
          </View>
        </View>

        {/* Login Button */}
        <TouchableOpacity 
          onPress={handleSignIn} 
          disabled={loading}
          style={{
            backgroundColor: '#0b8c5c',
            paddingVertical: 12,
            borderRadius: 8,
            alignItems: 'center',
            marginTop: 32,
            marginBottom: 48,
            opacity: loading ? 0.7 : 1
          }}
        >
          {loading ? (
            <ActivityIndicator color="white" size="small" />
          ) : (
            <Text style={{ color: 'white', fontSize: 18, fontWeight: '600' }}>Login</Text>
          )}
        </TouchableOpacity>

        {/* Don't have an account? */}
        <View style={{ flexDirection: 'row', justifyContent: 'center', marginBottom: 32 }}>
          <Text style={{ color: '#4b5563' }}>Don't have an account? </Text>
          <TouchableOpacity onPress={() => router.push('(auth)/sign-up')}>
            <Text style={{ color: '#0b8c5c', fontWeight: '600' }}>Sign Up</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
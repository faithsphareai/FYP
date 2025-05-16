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

export default function SignUpScreen() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  // Check if user is already logged in
  useEffect(() => {
    if (isAuthenticated()) {
      router.replace('(root)/(screens)/home');
    }
  }, [isAuthenticated, router]);

  const handleSignUp = async () => {
    try {
      // Validate inputs
      if (!username || !email || !password) {
        Alert.alert('Error', 'Please fill in all fields');
        return;
      }
      
      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        Alert.alert('Error', 'Please enter a valid email address');
        return;
      }
      
      // Password validation (at least 6 characters)
      if (password.length < 6) {
        Alert.alert('Error', 'Password must be at least 6 characters long');
        return;
      }
      
      setLoading(true);
      
      // Use authService instead of database functions
      const response = await authService.signup(username, email, password);
      
      setLoading(false);
      
      if (!response) {
        Alert.alert('Error', 'Failed to create account. Please try again.');
        return;
      }

      // Reset form
      setUsername('');
      setEmail('');
      setPassword('');
      
      Alert.alert(
        'Success', 
        'Account created successfully!',
        [{ text: 'OK', onPress: () => router.push('(auth)/sign-in') }]
      );
    } catch (error) {
      setLoading(false);
      
      let errorMessage = 'Failed to create account. Please try again.';
      
      if (error.response && error.response.data && error.response.data.message) {
        errorMessage = error.response.data.message;
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
        {/* Signup Image */}
        <View style={{ alignItems: 'center', marginTop: 80 }}>
          <Image 
            source={images.signup} 
            resizeMode='contain'
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
            Create an Account
          </Text>

          {/* Username Field */}
          <View style={{ borderBottomWidth: 1, borderBottomColor: '#d1d5db', paddingBottom: 8 }}>
            <TextInput
              placeholder="Username"
              placeholderTextColor="#A0AEC0"
              style={{ fontSize: 18, color: '#4b5563' }}
              value={username}
              onChangeText={setUsername}
            />
          </View>

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

        {/* Signup Button */}
        <TouchableOpacity 
          onPress={handleSignUp} 
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
            <Text style={{ color: 'white', fontSize: 18, fontWeight: '600' }}>Sign Up</Text>
          )}
        </TouchableOpacity>

        {/* Already have an account? */}
        <View style={{ flexDirection: 'row', justifyContent: 'center', marginBottom: 32 }}>
          <Text style={{ color: '#4b5563' }}>Already have an account? </Text>
          <TouchableOpacity onPress={() => router.push('(auth)/sign-in')}>
            <Text style={{ color: '#0b8c5c', fontWeight: '600' }}>Sign In</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
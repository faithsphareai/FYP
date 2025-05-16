import React, { useState } from 'react';
import { View, Text, ScrollView, Image, Dimensions, StyleSheet, StatusBar } from 'react-native';
import { useRouter } from 'expo-router';
import { TouchableOpacity } from 'react-native-gesture-handler';
import * as Animatable from 'react-native-animatable';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../../../context/AuthContext';
import { useEffect } from 'react';
import authService from '../../../utils/services/authService';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback } from 'react';

export default function HomeScreen() {
  const router = useRouter();
  const { user, logout, refreshUserData } = useAuth();
  const screenWidth = Dimensions.get('window').width;
  const buttonSize = (screenWidth - 48) / 2;
  const [userAvatarUrl, setUserAvatarUrl] = useState(null);

  // Add session validation on component mount
  useEffect(() => {
    validateSession();
  }, []);

  // Add focus effect to refresh user data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      const refreshData = async () => {
        if (user) {
          // Refresh user data from the server
          await refreshUserData();
          // Then load the avatar
          loadUserAvatar();
        }
      };
      
      refreshData();
    }, [user])
  );

  const loadUserAvatar = async () => {
    if (user && user.avatar) {
      try {
        const url = await authService.getAvatarUrl(user.avatar);
        setUserAvatarUrl(url);
      } catch (error) {
        console.log('Error loading avatar:', error);
      }
    }
  };

  useEffect(() => {
    loadUserAvatar();
  }, [user]);

  // Function to validate user session
  const validateSession = async () => {
    try {
      // If no user is logged in, no need to validate
      if (!user) return;

      // Try to get user data to check if session is valid
      const userData = await authService.getUserData();

      // If we can't get user data, session might be expired
      if (!userData) {
        console.log('Session expired or invalid');
        // Logout the user
        await logout();
      }
    } catch (error) {
      // If we get a 401 error, the token is invalid/expired
      if (error.response && error.response.status === 401) {
        console.log('Session expired');
        // Logout the user
        await logout();
      }
    }
  };

  const buttons = [
    {
      title: 'Recitation',
      route: '/recitation',
      image: require('../../../assets/recitationVector.jpg'),
    },
    {
      title: 'OCR Scanner',
      route: '/ocr',
      image: require('../../../assets/quranAuthVector.jpg'),
    },
    {
      title: 'Namaz Tariqa',
      route: '/namaz-tariqa',
      image: require('../../../assets/namazVector.jpg'),
    },
    {
      title: 'Quiz',
      route: {
        pathname: '/chatbot',
        params: {
          chatContext: 'quiz',
          title: 'Islamic Quiz'
        }
      },
      image: require('../../../assets/quizVector.jpg'),
    },
    {
      title: 'Hadith Auth',
      route: {
        pathname: '/chatbot',
        params: {
          chatContext: 'hadith',
          title: 'Hadith Authentication'
        }
      },
      image: require('../../../assets/hadithAuthVector.jpg'),
    },
    {
      title: 'Quran Auth',
      route: {
        pathname: '/chatbot',
        params: {
          chatContext: 'quran',
          title: 'Quran Ayat Authentication'
        }
      },
      image: require('../../../assets/quranAuthVector.jpg'),
    },
    {
      title: 'General Chatbot',
      route: {
        pathname: '/chatbot',
        params: {
          chatContext: 'generalChatbot',
          title: 'General Chatbot'
        }
      },
      image: require('../../../assets/messages.png'),
    },
    {
      title: 'Monthly Prayer Timings',
      route:"/monthly-prayer-timings",
      image: require('../../../assets/islamicCalander.png'),
    },
    {
      title: 'Get Nearby Mosques',
      route:"/get-nearby-mosques",
      image: require('../../../assets/mosqueMap.png'),
    },
  ];

  // Update the handleButtonPress function to include profile navigation
  const handleButtonPress = (route) => {
    // Check if it's a protected route (chatbot or ocr)
    if (
      (typeof route === 'string' && (route === '/ocr' || route === '/profile')) ||
      (typeof route === 'object' && route.pathname === '/chatbot')
    ) {
      if (!user) {
        // Redirect to login if user is not authenticated
        router.push('/(auth)/sign-in');
        return;
      }
    }
    router.push(route);
  };

  

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <View style={styles.headerImageContainer}>
        <Animatable.View animation="fadeIn" duration={1000} style={styles.headerImageWrapper}>
          <Image
            source={require('../../../assets/home.jpg')}
            style={styles.headerImage}
          />
          <LinearGradient
            colors={['rgba(0,0,0,0.3)', 'rgba(0,0,0,0)']}
            style={styles.headerGradient}
          />
        </Animatable.View>

        <Animatable.View animation="fadeIn" delay={300} style={styles.headerTextContainer}>
          <Text style={styles.titleText}>FaithSphere</Text>
          <Text style={styles.subtitleText}>
            Your one-stop destination for spiritual growth and knowledge.
          </Text>
        </Animatable.View>
      </View>

      <Animatable.View animation="fadeInUp" delay={500} style={styles.welcomeCard}>
        <TouchableOpacity 
          activeOpacity={0.8}
          onPress={user ? () => handleButtonPress('/profile') : () => router.push('/(auth)/sign-in')}
        >
          <View style={styles.welcomeCardContent}>
            <View style={styles.welcomeCardHeader}>
              <View style={styles.welcomeTextContainer}>
                <Text style={styles.welcomeText}>
                  Welcome, <Text style={styles.nameText}>{user ? user.name : 'Guest'}</Text>!
                </Text>
                <Text style={styles.welcomeSubtext}>
                  {user ? "Continue your spiritual journey" : "Sign in to access all features"}
                </Text>
              </View>
              
              {user ? (
                <View style={styles.welcomeAvatarContainer}>
                  {userAvatarUrl ? (
                    <Image
                      source={{ uri: userAvatarUrl }}
                      style={styles.welcomeAvatar}
                    />
                  ) : (
                    <View style={styles.welcomeAvatarPlaceholder}>
                      <Text style={styles.welcomeAvatarText}>
                        {user.name ? user.name.charAt(0).toUpperCase() : 'U'}
                      </Text>
                    </View>
                  )}
                </View>
              ) : (
                <View style={styles.welcomeLoginIcon}>
                  <Ionicons name="log-in-outline" size={22} color="#0b8c5c" />
                </View>
              )}
            </View>
            
            <View style={styles.welcomeCardFooter}>
              <Text style={styles.welcomeCardAction}>
                {user ? "View Profile" : "Sign In"}
              </Text>
              <Ionicons name="chevron-forward" size={16} color="#0b8c5c" />
            </View>
          </View>
        </TouchableOpacity>
      </Animatable.View>

      <ScrollView style={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        <Text style={styles.sectionTitle}>Explore Features</Text>

        <View style={styles.buttonGrid}>
          {buttons.map((button, index) => (
            <Animatable.View
              key={button.title}
              animation="fadeInUp"
              delay={index * 100}
              style={{
                width: buttonSize,
                height: buttonSize,
                marginBottom: 16,
              }}
            >
              <TouchableOpacity
                onPress={() => handleButtonPress(button.route)}
                style={[{
                  width: '100%',
                  height: '100%',
                }, styles.buttonContainer]}
              >
                <Image
                  source={button.image}
                  style={styles.buttonBackgroundImage}
                  resizeMode="cover"
                />
                <LinearGradient
                  colors={['rgba(0,0,0,0)', 'rgba(0,0,0,0.7)']}
                  style={styles.buttonGradient}
                />
                <View style={styles.buttonTextOverlay}>
                  <Text style={styles.buttonText}>
                    {button.title}
                  </Text>
                </View>
              </TouchableOpacity>
            </Animatable.View>
          ))}

        </View>
      </ScrollView>
     
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
  headerImageContainer: {
    position: 'relative',
    width: '100%',
  },
  headerImageWrapper: {
    width: '100%',
    position: 'relative',
  },
  headerImage: {
    width: '100%',
    height: Dimensions.get('window').width * 0.6,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  headerGradient: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '100%',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  headerTextContainer: {
    position: 'absolute',
    top: 20,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
  },
  // Add these new styles to your StyleSheet
  bottomProfileContainer: {
    position: 'absolute',
    bottom: 20,
    left: 16,
    right: 16,
    zIndex: 10,
  },
  bottomProfileCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  bottomProfileContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
  },
  bottomProfileAvatarContainer: {
    marginRight: 12,
  },
  bottomProfileAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#f3f4f6',
  },
  bottomProfileAvatarPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#0b8c5c',
    justifyContent: 'center',
    alignItems: 'center',
  },
  bottomProfileAvatarText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  bottomProfileInfo: {
    flex: 1,
  },
  bottomProfileName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
  },
  bottomProfileSubtext: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  bottomProfileArrow: {
    padding: 4,
  },
  bottomLoginCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  bottomLoginContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
  },
  bottomLoginIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f0fdf4',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  bottomLoginInfo: {
    flex: 1,
  },
  bottomLoginText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1f2937',
  },
  bottomLoginSubtext: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  bottomLoginArrow: {
    padding: 4,
  },
  titleText: {
    textAlign: 'center',
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  subtitleText: {
    textAlign: 'center',
    fontSize: 14,
    color: 'white',
    marginTop: 8,
    marginHorizontal: 16,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  // Update these existing styles
  welcomeCard: {
    marginHorizontal: 16,
    marginTop: -20,
    backgroundColor: 'white',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    overflow: 'hidden',
  },
  welcomeCardContent: {
    padding: 16,
  },
  // Add these new styles
  welcomeCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  welcomeTextContainer: {
    flex: 1,
  },
  welcomeAvatarContainer: {
    marginLeft: 12,
  },
  welcomeAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#f3f4f6',
  },
  welcomeAvatarPlaceholder: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#0b8c5c',
    justifyContent: 'center',
    alignItems: 'center',
  },
  welcomeAvatarText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  welcomeLoginIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#f0fdf4',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 12,
  },
  welcomeCardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  welcomeCardAction: {
    fontSize: 14,
    fontWeight: '500',
    color: '#0b8c5c',
    marginRight: 4,
  },
  welcomeText: {
    fontSize: 16,
    color: '#374151',
  },
  nameText: {
    fontWeight: 'bold',
    color: '#15803d',
  },
  welcomeSubtext: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
    marginTop: 24,
    marginBottom: 16,
    paddingHorizontal: 16,
  },
  scrollContainer: {
    flex: 1,
  },
  buttonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 20,
  },
  // Update these button styles
  buttonContainer: {
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 3,
  },
  buttonBackgroundImage: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: 12,
  },
  buttonGradient: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '50%',
    borderBottomLeftRadius: 12,
    borderBottomRightRadius: 12,
  },
  buttonTextOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 12,
    justifyContent: 'flex-end',
  },
  buttonText: {
    textAlign: 'center',
    color: 'white',
    fontWeight: '600',
    fontSize: 14,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  profileButtonContainer: {
    borderRadius: 12,
    overflow: 'hidden',
  },
  profileButtonGradient: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileButtonContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileButtonText: {
    color: 'white',
    fontWeight: '600',
    marginTop: 8,
    fontSize: 16,
  },
  // Add these styles to the StyleSheet
  profileCardContainer: {
    marginHorizontal: 16,
    marginTop: 8,
    marginBottom: 24,
  },
  profileCard: {
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 4,
  },
  profileCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  profileAvatarContainer: {
    marginRight: 12,
  },
  profileAvatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 2,
    borderColor: 'white',
  },
  profileAvatarPlaceholder: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'white',
  },
  profileAvatarText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  profileInfoContainer: {
    flex: 1,
  },
  profileName: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  profileEmail: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    marginTop: 2,
  },
  profileIconContainer: {
    padding: 4,
  },
  loginCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  loginIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  loginTextContainer: {
    flex: 1,
  },
  loginText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loginSubtext: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    marginTop: 2,
  },
  loginArrowContainer: {
    padding: 4,
  },
});
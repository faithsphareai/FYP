import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import authService from '../utils/services/authService';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for existing user session when app loads
  useEffect(() => {
    checkUser();
  }, []);

  const login = async (userData) => {
    try {
      await AsyncStorage.setItem('user', JSON.stringify(userData));
      setUser(userData);
      return true;
    } catch (error) {
      console.error('Error logging in:', error);
      return false;
    }
  };

  const logout = async () => {
    try {
      await AsyncStorage.removeItem('user');
      setUser(null);
      return true;
    } catch (error) {
      console.error('Error logging out:', error);
      return false;
    }
  };

  const checkUser = async () => {
    try {
      const userData = await AsyncStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
      }
      setLoading(false);
    } catch (error) {
      console.error('Error checking user:', error);
      setLoading(false);
    }
  };

  const isAuthenticated = () => {
    return user !== null;
  };

  // Add this function to your AuthContext provider
  const refreshUserData = async () => {
    try {
      // Get fresh user data from the server
      const userData = await authService.getUserData();
      if (userData) {
        // Update the user state with fresh data
        setUser(userData);
        // Also update AsyncStorage
        await AsyncStorage.setItem('user', JSON.stringify(userData));
        return userData;
      }
    } catch (error) {
      console.log('Error refreshing user data:', error);
    }
    return null;
  };
  
  return (
    <AuthContext.Provider value={{ 
      user, 
      login, 
      logout, 
      loading, 
      checkUser,
      isAuthenticated,
      refreshUserData // Add the refreshUserData function to the context
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
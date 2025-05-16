import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

const API_URL = 'https://hammad712-auth.hf.space/auth';
const API_AVATAR_URL = 'https://hammad712-auth.hf.space';


export const authService = {
  async signup(name, email, password) {
    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('email', email);
      formData.append('password', password);
      
      const response = await axios.post(`${API_URL}/signup`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Accept': 'application/json'
        }
      });
      
      if (response.status === 200) {
        // Store tokens in secure storage
        await storeTokens(response.data);
        return response.data;
      }
      
      return null;
    } catch (error) {
      console.error('Signup error:', error);
      throw error;
    }
  },
  
  async login(email, password) {
    try {
      // Create URLSearchParams object for form data
      const formData = new URLSearchParams();
      formData.append('grant_type', 'password');
      formData.append('username', email);
      formData.append('password', password);
      formData.append('scope', '');
      formData.append('client_id', 'string');
      formData.append('client_secret', 'string');
      
      const response = await axios.post(`${API_URL}/login`, formData.toString(), {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      
      if (response.status === 200) {
        // Store tokens in secure storage
        await storeTokens(response.data);
        
        // Store email for future reference
        await SecureStore.setItemAsync('email', email);
        
        // Optionally store password for re-authentication scenarios
        // Note: Storing passwords is generally not recommended for security reasons
        // but in this case it helps with the re-authentication flow
        await SecureStore.setItemAsync('password', password);
        
        return response.data;
      }
      
      return null;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },
  
  async getUser() {
    try {
      const token = await SecureStore.getItemAsync('access_token');
      
      if (!token) {
        return null;
      }
      
      return {
        name: await SecureStore.getItemAsync('name') || '',
        avatar: await SecureStore.getItemAsync('avatar') || null
      };
    } catch (error) {
      return null;
    }
  },
  
  async getUserData() {
    try {
      const token = await SecureStore.getItemAsync('access_token');
      const tokenType = await SecureStore.getItemAsync('token_type');
      
      if (!token) {
        return null;
      }
      
      const response = await axios.get(`${API_URL}/user/data`, {
        headers: {
          'Accept': 'application/json',
          'Authorization': `${tokenType} ${token}`
        }
      });
      
      if (response.status === 200) {
        return response.data;
      }
      
      return null;
    } catch (error) {
      throw error;
    }
  },
  async updateUser(userData) {
    try {
      const token = await SecureStore.getItemAsync('access_token');
      const tokenType = await SecureStore.getItemAsync('token_type');
      
      if (!token) {
        return null;
      }
      
      const formData = new FormData();
      if (userData.name) formData.append('name', userData.name);
      if (userData.email) formData.append('email', userData.email);
      if (userData.password) formData.append('password', userData.password);
      
      // Handle avatar image upload
      if (userData.avatar) {
        // Create file object from uri
        const fileUri = userData.avatar;
        const filename = fileUri.split('/').pop();
        const match = /\.(\w+)$/.exec(filename);
        const type = match ? `image/${match[1]}` : 'image/jpeg';
        
        formData.append('avatar', {
          uri: fileUri,
          name: filename,
          type
        });
      }
      
      const response = await axios.put(`${API_URL}/user/update`, formData, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'multipart/form-data',
          'Authorization': `${tokenType} ${token}`
        }
      });
      
      if (response.status === 200) {
        // Update stored user info
        if (userData.name) {
          await SecureStore.setItemAsync('name', userData.name);
        }
        
        // If avatar was updated, store the avatar ID
        if (response.data.avatar) {
          await SecureStore.setItemAsync('avatar', String(response.data.avatar));
        }
        
        // If email was updated, we need to re-login to get new tokens
        if (userData.email) {
          // Store the new email
          await SecureStore.setItemAsync('email', userData.email);
          
          // If we have the password, we can automatically re-login
          const storedPassword = await SecureStore.getItemAsync('password');
          console.log(storedPassword)
          if (storedPassword) {
            try {
              // Re-login with new email
              const loginResponse = await this.login(userData.email, storedPassword);
              if (loginResponse) {
                console.log('Re-authenticated with new email');
              }
            } catch (loginError) {
              // We'll return success anyway since the update worked
            }
          } else {
            // We don't have the password stored, so we can't auto re-login
            // The user will need to log out and log back in
          }
        }
        
        return response.data;
      }
      
      return null;
    } catch (error) {
      console.log('Update user error:', error);
      throw error;
    }
  },
  
  async getAvatarUrl(avatarId) {
    if (!avatarId) return null;
    
    try {
      const token = await SecureStore.getItemAsync('access_token');
      
      if (!token) {
        return null;
      }
      
      // Return the full URL to the avatar with optional timestamp to prevent caching
      let url = `${API_AVATAR_URL}${avatarId}`;
      
      
      return url;
    } catch (error) {
      console.log('Get avatar URL error:', error);
      return null;
    }
  },
  
  
  async logout() {
    try {
      await SecureStore.deleteItemAsync('access_token');
      await SecureStore.deleteItemAsync('refresh_token');
      await SecureStore.deleteItemAsync('token_type');
      await SecureStore.deleteItemAsync('name');
      await SecureStore.deleteItemAsync('avatar');
      return true;
    } catch (error) {
      console.log('Logout error:', error);
      return false;
    }
  }

};

// Helper function to store tokens
const storeTokens = async (tokenData) => {
  try {
    await SecureStore.setItemAsync('access_token', tokenData.access_token);
    await SecureStore.setItemAsync('refresh_token', tokenData.refresh_token);
    await SecureStore.setItemAsync('token_type', tokenData.token_type);
    
    // Store user info if available
    if (tokenData.name) {
      await SecureStore.setItemAsync('name', tokenData.name);
    }
    
    if (tokenData.avatar) {
      await SecureStore.setItemAsync('avatar', String(tokenData.avatar));
    }
  } catch (error) {
    console.error('Error storing tokens:', error);
  }
};

export default authService;


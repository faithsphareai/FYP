// utils/services/apis.js

import axios from 'axios';

const BASE_URL = 'https://hammad712-urdu-ocr-app.hf.space';
const BASE_URL2 = 'https://hammad712-recitation-compare.hf.space';

// Ayat Finder API URLs
const AYAT_FINDER_URL_ENGLISH= 'https://hammad712-ayat-finder.hf.space/query';
const AYAT_FINDER_URL_ARABIC = 'https://hammad712-arabic-ayat.hf.space/query';

// Add new constant for Hadith Finder API
const HADITH_FINDER_URL_ENGLISH = 'https://hammad712-hadith-finder.hf.space/query';
const HADITH_FINDER_URL_ARABIC = 'https://hammad712-arabic-hadith.hf.space/query';


const API_ENDPOINTS = {
  OCR: '/upload',
  COMPARE_DTW: '/compare-dtw'
};

// Helper function for error handling
const handleApiError = (error, service) => {
  if (error.response) {
    return new Error(
      `${service} server error: ${error.response.status} - ${error.response.data.message || 'Unknown error'}`
    );
  } else if (error.request) {
    return new Error(`No response from ${service} service. Please check your internet connection.`);
  } else {
    return new Error(`${service} request failed: ${error.message}`);
  }
};


export const extractTextFromFile = async (fileUri, fileType) => {
  try {
    const formData = new FormData();
    const fileName = fileUri.split('/').pop();
    const mimeType = fileType === 'pdf' ? 'application/pdf' : 'image/jpeg';
    
    formData.append('file', {
      uri: fileUri,
      name: fileName,
      type: mimeType,
    });

    const response = await axios.post(`${BASE_URL}${API_ENDPOINTS.OCR}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000,
    });

    return response.data.extracted_text;
    
  } catch (error) {
    console.error('OCR API Error:', error);
    if (error.response) {
      throw new Error(`Server error: ${error.response.status} - ${error.response.data.message || 'Unknown error'}`);
    } else if (error.request) {
      throw new Error('No response from OCR service. Please check your internet connection.');
    } else {
      throw new Error(`OCR request failed: ${error.message}`);
    }
  }
};

// Update the compareDTW function to match the API expectations
export const compareDTW = async (originalAudio, userAudio) => {
  try {
    const formData = new FormData();
    
    // Add file extension check and conversion if needed
    const originalExt = originalAudio.name.split('.').pop().toLowerCase();
    const userExt = userAudio.name.split('.').pop().toLowerCase();
    
    console.log(`File extensions: original=${originalExt}, user=${userExt}`);
    
    formData.append('audio1', {
      uri: originalAudio.uri,
      name: originalAudio.name,
      type: originalExt === 'mp3' ? 'audio/mp3' : 'audio/ogg',
    });

    formData.append('audio2', {
      uri: userAudio.uri,
      name: userAudio.name,
      type: userExt === 'mp3' ? 'audio/mp3' : 'audio/ogg',
    });

    console.log('Making API request to:', `${BASE_URL2}${API_ENDPOINTS.COMPARE_DTW}`);
    
    const response = await axios.post(`${BASE_URL2}${API_ENDPOINTS.COMPARE_DTW}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Accept': 'application/json',
      },
      timeout: 160000, 
    });
    
    
      return response.data;
    
    
  } catch (error) {
    console.error('DTW Comparison API Error:', error);
    
    // More detailed error logging
    if (error.response) {
      console.error('Error response data:', error.response.data);
      throw new Error(`Server error: ${error.response.status} - ${error.response.data.message || 'Unknown error'}`);
    } else if (error.request) {
      console.error('Error request details:', error.request);
    } 
  }
};

/**
 * Sends a question to the Ayat Finder API and returns the answer.
 * @param {string} question
 * @returns {Promise<string>} The answer from the API.
 */
export const getAyatAnswer = async (question, language) => {
  const url = language === 'ar' ? AYAT_FINDER_URL_ARABIC : AYAT_FINDER_URL_ENGLISH;

  console.log("url using", url)
  console.log("language", language)

  try {
    const response = await axios.post(
      url,
      { question },
      {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 30000,
      }
    );
    // The API returns { answer: "..." }
    return response.data.answer;
  } catch (error) {
    console.error('Ayat Finder API Error:', error);
    if (error.response) {
      throw new Error(
        `Server error: ${error.response.status} - ${error.response.data.message || 'Unknown error'}`
      );
    } else if (error.request) {
      throw new Error('No response from Ayat Finder service. Please check your internet connection.');
    } else {
      throw new Error(`Ayat Finder request failed: ${error.message}`);
    }
  }
};

const QUIZ_API_URL = 'https://hammad712-islamic-quiz.hf.space/quiz';
const GRADE_QUIZ_API_URL = 'https://hammad712-islamic-quiz.hf.space/grade';

/**
 * Generates a quiz question from the API
 * @param {string} search_query - The topic or context for the quiz
 * @returns {Promise<string>} The quiz question
*/

export const generateQuiz = async (search_query) => {
  console.log("search query", search_query);
  try {
    const response = await axios.post(
      QUIZ_API_URL,
      {"question": search_query },
      {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 30000,
      }
    );
   
    console.log("gen quiz axiossss error", response);
    return response.data.quiz;
  } catch (error) {
    console.error('Quiz Generation API Error:', error);
    throw handleApiError(error, 'Quiz Generation');
  }
};

/**
 * Grades the user's answer to a quiz question
 * @param {string} answers - The user's answer
 * @returns {Promise<string>} The grading result
 */
export const gradeQuiz = async (answers) => {
  try {
    const response = await axios.post(
      GRADE_QUIZ_API_URL,
      { answers },
      {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 30000,
      }
    );
    return response.data.grade;
  } catch (error) {
    console.error('Quiz Grading API Error:', error);
    throw handleApiError(error, 'Quiz Grading');
  }
};


/**
 * Sends a question to the Hadith Finder API and returns the answer
 * @param {string} question - The hadith query
 * @returns {Promise<string>} The answer from the API
 */
export const getHadithAnswer = async (question, language) => {
  const url = language === 'ar' ? HADITH_FINDER_URL_ARABIC : HADITH_FINDER_URL_ENGLISH;

  console.log("url using", url);
  console.log("language", language);
  
  try {
    const response = await axios.post(
      url,
      { question },
      {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 30000,
      }
    );
    
    if (!response.data || !response.data.answer) {
      throw new Error('Invalid response format from Hadith Finder API');
    }
    
    return response.data.answer;
  } catch (error) {
    console.error('Hadith Finder API Error:', error);
    throw handleApiError(error, 'Hadith Finder');
  }
};


const GENERAL_CHATBOT_URL = 'https://hammad712-chatbot.hf.space/query';
/**
 * Sends a question to the General Chatbot API and returns the answer
 * @param {string} question - The query
 * @returns {Promise<string>} The answer from the API
 */
export const getGeneralChatbotAnswer = async (question, language) => {
 
  try {
    const response = await axios.post(
      GENERAL_CHATBOT_URL,
      { question },
      {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        timeout: 30000,
      }
    );
    
    if (!response.data || !response.data.answer) {
      throw new Error('Invalid response format from Genral Chatbot API');
    }
    
    return response.data.answer;
  } catch (error) {
    console.error('Genral Chatbot API Error:', error);
    throw handleApiError(error, 'Genral Chatbot');
  }
};


// Add new constant for Aladhan API
const ALADHAN_API_URL = 'https://api.aladhan.com/v1/calendar';

/**
 * Fetches prayer times from the Aladhan API based on coordinates and date
 * @param {Object} coords - The coordinates object containing latitude and longitude
 * @param {number} year - The year for which prayer times are needed
 * @param {number} month - The month (1-12) for which prayer times are needed
 * @param {number} method - The calculation method (default is 1)
 * @param {number} school - The school of thought (0 or 1)
 * @returns {Promise<Object>} The prayer times calendar data
 */
export const getPrayerCalendar = async (coords, year, month, method = 1, school = 0) => {
  try {
    const response = await axios.get(`${ALADHAN_API_URL}/${year}/${month}`, {
      params: {
        latitude: coords.coords.latitude,
        longitude: coords.coords.longitude,
        method,
        school,
      },
      timeout: 10000,
    });

    if (!response.data || response.data.code !== 200 || !response.data.data) {
      throw new Error('Invalid response format from Aladhan API');
    }

    return response.data.data;
  } catch (error) {
    console.error('Aladhan API Error:', error);
    throw handleApiError(error, 'Aladhan');
  }
};

// Constants
const RAPIDAPI_MAP_URL = 'https://maps-data.p.rapidapi.com/nearby.php';

/**
 * Fetches nearby mosques using the RapidAPI Maps Data API
 * @param {Object} coords - Coordinates object containing latitude and longitude
 * @param {number} limit - Maximum number of results to return (default: 10)
 * @param {number} zoom - Map zoom level (default: 10)
 * @returns {Promise<Array>} Array of nearby mosque data
 */
export const getNearByMosques = async (coords, limit = 10, zoom = 10) => {
  try {
    const response = await axios.get(RAPIDAPI_MAP_URL, {
      params: {
        query: 'masjid',
        lat: coords.coords.latitude,
        lng: coords.coords.longitude,
        limit,
        zoom,
      },
      headers: {
        'X-RapidAPI-Key': 'dbbd245e5fmsh2db3402603f064cp14156bjsnd0e7bd00dc0a',
        'X-RapidAPI-Host': 'maps-data.p.rapidapi.com'
      },
      timeout: 10000
    });

    if (!response.data?.data) {
      throw new Error('Invalid response format from Maps Data API');
    }

    return response.data.data.map(item => ({
      name: item.name,
      lat: item.latitude,
      long: item.longitude,
      fullAddress: item.full_address,
      photos: item.photos || [],
      rating: item.rating,
      openNow: item.open_now
    }));
    
  } catch (error) {
    console.error('Maps Data API Error:', error);
    throw handleApiError(error, 'Maps Data');
  }
};
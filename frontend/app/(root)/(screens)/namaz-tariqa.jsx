import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Image, SafeAreaView } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

export default function PrayerScreen() {
  const { prayer } = useLocalSearchParams();
  const [gender, setGender] = useState('male');
  const [currentStep, setCurrentStep] = useState(1);
  const router = useRouter();
  const scrollViewRef = useRef();
  const [prayerType, setPrayerType] = useState('Fajr');
  
  // Add this useEffect hook to ensure consistent hook order
  useEffect(() => {
    // You can put any initialization logic here
    // This ensures the hook is always called
  }, []);

  // Define prayer steps for demonstration
  const prayerSteps = {
    Fajr: {
      male: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Raise your hands to your ears and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ],
      female: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Women should raise their hands to shoulder level and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg') // Use women-specific images if available
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Women should keep their arms close to their body. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Women should bend less deeply than men. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ]
    },
    Dhuhr: {
      male: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Raise your hands to your ears and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ],
      female: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Women should raise their hands to shoulder level and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Women should keep their arms close to their body. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Women should bend less deeply than men. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ]
    },
    Asr: {
      male: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Raise your hands to your ears and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ],
      female: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Women should raise their hands to shoulder level and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Women should keep their arms close to their body. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Women should bend less deeply than men. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ]
    },
    Maghrib: {
      male: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Raise your hands to your ears and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ],
      female: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Women should raise their hands to shoulder level and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Women should keep their arms close to their body. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Women should bend less deeply than men. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ]
    },
    Isha: {
      male: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Raise your hands to your ears and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ],
      female: [
        {
          title: 'Standing Position (Qiyam)',
          description: 'Stand straight facing the Qibla. Women should raise their hands to shoulder level and say "Allahu Akbar".',
          arabicText: 'الله أكبر',
          translation: 'Allahu Akbar',
          meaning: 'Allah is the Greatest',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Recitation Position',
          description: 'Place your right hand over your left hand on your chest. Women should keep their arms close to their body. Recite Surah Al-Fatiha and another short Surah.',
          arabicText: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
          translation: 'Bismillahir Rahmanir Raheem',
          meaning: 'In the name of Allah, the Most Gracious, the Most Merciful',
          image: require('../../../assets/qiyam.jpg')
        },
        {
          title: 'Bowing Position (Ruku)',
          description: 'Bend forward at the waist, placing hands on knees. Women should bend less deeply than men. Say "Subhana Rabbiyal Azeem" three times.',
          arabicText: 'سُبْحَانَ رَبِّيَ الْعَظِيمِ',
          translation: 'Subhana Rabbiyal Azeem',
          meaning: 'Glory be to my Lord, the Most Great',
          image: require('../../../assets/ruku.jpg')
        }
      ]
    }
  };

  // Get steps for the current prayer or use an empty array if not found
  const steps = prayerSteps[prayerType] && prayerSteps[prayerType][gender] 
    ? prayerSteps[prayerType][gender] 
    : [];
  const totalSteps = steps.length;
  
  // Ensure currentStep is valid
  useEffect(() => {
    if (currentStep > totalSteps && totalSteps > 0) {
      setCurrentStep(1);
    }
  }, [prayerType, gender, totalSteps, currentStep]);

  const handleBack = () => {
    router.back();
  };

  const goToNextStep = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
      scrollViewRef.current?.scrollTo({ y: 0, animated: true });
    }
  };

  const goToPreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      scrollViewRef.current?.scrollTo({ y: 0, animated: true });
    }
  };

  const renderPrayerTabs = () => {
    const prayers = ['Fajr', 'Dhuhr', 'Asr', 'Maghrib', 'Isha'];
    
    return (
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.prayerTabs}>
        
        {prayers.map((prayerName) => (
          <TouchableOpacity
            key={prayerName}
            onPress={() => {
              setCurrentStep(1); 
              setPrayerType(prayerName);
            }}
            style={[
              styles.prayerTab,
              { backgroundColor: prayerName === prayerType ? '#15803d' : 'white' }
            ]}
          >
            <Text 
              style={[
                styles.prayerTabText, 
                { color: prayerName === prayerType ? 'white' : '#333' }
              ]}
            >
              {prayerName}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    );
  };

  // Get current step data safely
  const currentStepData = steps.length > 0 && currentStep <= steps.length 
    ? steps[currentStep - 1] 
    : {
        title: '',
        description: '',
        arabicText: '',
        translation: '',
        meaning: '',
        image: null
      };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#15803d" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>
          Namaz Tariqa
        </Text>
        <View style={styles.headerSpace} />
      </View>

      <View style={styles.introContainer}>
        <Text style={styles.introText}>
          Learn how to perform prayers correctly with step-by-step visual instructions.
        </Text>
      </View>
    
      <View style={styles.prayerTabsContainer}>
        {renderPrayerTabs()}
      </View>
      
      {/* Gender toggle buttons */}
      <View style={styles.genderToggleContainer}>
        <TouchableOpacity 
          style={[
            styles.genderButton, 
            gender === 'male' ? styles.genderButtonActive : null
          ]}
          onPress={() => setGender('male')}
        >
          <Text style={[
            styles.genderButtonText,
            gender === 'male' ? styles.genderButtonTextActive : null
          ]}>Men</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[
            styles.genderButton, 
            gender === 'female' ? styles.genderButtonActive : null
          ]}
          onPress={() => setGender('female')}
        >
          <Text style={[
            styles.genderButtonText,
            gender === 'female' ? styles.genderButtonTextActive : null
          ]}>Women</Text>
        </TouchableOpacity>
      </View>

      <ScrollView 
        ref={scrollViewRef}
        style={styles.scrollView}
        contentContainerStyle={styles.scrollViewContent}
        showsVerticalScrollIndicator={false}
      >
        {steps.length > 0 ? (
          <View style={styles.stepContainer}>
            <Text style={styles.prayerTitle}>
              {prayerType} Prayer {gender === 'female' ? '(Women)' : ''}
            </Text>
            <Text style={styles.stepIndicator}>
              Step {currentStep} of {totalSteps}
            </Text>
            
            <View style={styles.progressBar}>
              <View 
                style={[
                  styles.progressFill, 
                  { width: `${(currentStep / totalSteps) * 100}%` }
                ]} 
              />
            </View>

            <Text style={styles.stepTitle}>
              {currentStepData.title}
            </Text>

            <Text style={styles.stepDescription}>
              {currentStepData.description}
            </Text>

            {currentStepData.image && (
              <View style={styles.imageContainer}>
                <Image 
                  source={currentStepData.image}
                  style={styles.stepImage}
                  resizeMode="cover"
                />
              </View>
            )}

            <Text style={styles.arabicText}>
              {currentStepData.arabicText}
            </Text>

            <Text style={styles.translationText}>
              {currentStepData.translation}
            </Text>

            <Text style={styles.meaningText}>
              {currentStepData.meaning}
            </Text>
          </View>
        ) : (
          <View style={styles.noContentContainer}>
            <Text style={styles.noContentText}>
              Prayer guide content is being prepared.
            </Text>
          </View>
        )}
      </ScrollView>

      <View style={styles.navigationButtons}>
        <TouchableOpacity 
          style={[styles.navButton, styles.prevButton, { opacity: currentStep > 1 ? 1 : 0.5 }]}
          onPress={goToPreviousStep}
          disabled={currentStep <= 1 || steps.length === 0}
        >
          <Ionicons name="chevron-back" size={20} color="#6b7280" />
          <Text style={styles.prevButtonText}>Previous</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={[styles.navButton, styles.nextButton, { opacity: currentStep < totalSteps ? 1 : 0.5 }]}
          onPress={goToNextStep}
          disabled={currentStep >= totalSteps || steps.length === 0}
        >
          <Text style={styles.nextButtonText}>Next</Text>
          <Ionicons name="chevron-forward" size={20} color="white" />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
    padding: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 20,
    borderBottomWidth: 1,
    backgroundColor: '#fff', 
    borderColor: '#e5e7eb', 
    marginTop:6
  },
  backButton: {
    padding: 4,
  },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    color: '#15803d', 
    fontWeight: '600',
    textAlign: 'center',
  },
  headerSpace: {
    width: 24,
  },
  introContainer: {
    padding: 16,
    paddingBottom: 12,
  },
  introText: {
    fontSize: 16,
    color: '#4b5563',
    lineHeight: 24,
  },
  prayerTabsContainer: {
    paddingVertical: 8,
  },
  prayerTabs: {
    paddingHorizontal: 16,
  },
  prayerTab: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 50,
    marginRight: 10,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    minWidth: 80,
    alignItems: 'center', // Center text
  },
  prayerTabText: {
    fontWeight: '500',
    fontSize: 15,
    textAlign: 'center',
  },
  scrollView: {
    flex: 1,
  },
  scrollViewContent: {
    padding: 16,
    paddingBottom: 24,
  },
  stepContainer: {
    alignItems: 'flex-start',
    width: '100%',
  },
  prayerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  stepIndicator: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 8,
  },
  progressBar: {
    height: 4,
    width: '100%',
    backgroundColor: '#e5e7eb',
    borderRadius: 2,
    marginBottom: 24,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#15803d',
    borderRadius: 2,
  },
  stepTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  stepDescription: {
    fontSize: 16,
    color: '#4b5563',
    lineHeight: 24,
    marginBottom: 20,
  },
  imageContainer: {
    width: '100%',
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 8,
  },
  stepImage: {
    width: '90%',
    height: 220,
  },
  arabicText: {
    fontSize: 28,
    color: '#1f2937',
    textAlign: 'right',
    width: '100%',
    marginBottom: 12,
    fontWeight: '500',
  },
  translationText: {
    fontSize: 18,
    fontStyle: 'italic',
    color: '#4b5563',
    marginBottom: 6,
  },
  meaningText: {
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 20,
  },
  navigationButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
  },
  navButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  prevButton: {
    backgroundColor: '#f3f4f6',
  },
  nextButton: {
    backgroundColor: '#15803d',
  },
  prevButtonText: {
    color: '#6b7280',
    fontWeight: '500',
    marginLeft: 4,
    fontSize: 16,
  },
  nextButtonText: {
    color: 'white',
    fontWeight: '500',
    marginRight: 4,
    fontSize: 16,
  },
  noContentContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    minHeight: 300,
  },
  noContentText: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
  },
  genderToggleContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 16,
    paddingHorizontal: 16,
  },
  genderButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    marginHorizontal: 8,
    borderRadius: 20,
  },
  genderButtonActive: {
    backgroundColor: '#15803d',
    borderColor: '#15803d',
  },
  genderButtonText: {
    fontSize: 14,
    color: '#4b5563',
    fontWeight: '500',
  },
  genderButtonTextActive: {
    color: 'white',
  },
});

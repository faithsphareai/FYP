import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function Fajr() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  // Keep your existing fajrSteps data
  const fajrSteps = [
    {
      title: "Sunnah First Rakat",
      description: "Make Niyyah (intention) for 2 Sunnah rakats of Fajr prayer",
      details: [
        "Face the Qibla",
        "Raise hands to ears",
        "Say 'Allahu Akbar'",
        "Place hands on chest (right over left)",
        "Recite Thana, Ta'awwuz, Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite any other Surah"
      ],
      image: images.niyyah
    },
    {
      title: "Complete First Rakat",
      description: "Perform Ruku and Sujood for first Rakat",
      details: [
        "Go into Ruku saying 'Allahu Akbar'",
        "Say 'Subhana Rabbiyal Azeem' 3 times",
        "Rise saying 'Sami Allahu Liman Hamidah'",
        "Perform first Sajdah saying 'Allahu Akbar'",
        "Say 'Subhana Rabbiyal Ala' 3 times",
        "Sit briefly between Sajdah",
        "Perform second Sajdah similarly"
      ],
      image: images.ruku
    },
    {
      title: "Sunnah Second Rakat",
      description: "Stand for second Rakat and complete prayer",
      details: [
        "Stand up for second Rakat",
        "Recite Bismillah and Surah Al-Fatiha",
        "Recite another Surah",
        "Perform Ruku and Sujood as before",
        "Sit for Tashahhud",
        "Recite At-Tahiyyat",
        "Complete with Salam"
      ],
      image: images.qiyam
    },
    {
      title: "Farz First Rakat",
      description: "Begin 2 Farz rakats of Fajr prayer",
      details: [
        "Make new Niyyah for Farz prayer",
        "Say Takbir-e-Tahreema",
        "Recite Thana, Ta'awwuz, Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood"
      ],
      image: images.takbeer
    },
    {
      title: "Farz Second Rakat",
      description: "Complete Farz prayer",
      details: [
        "Stand for second Rakat",
        "Recite Surah Al-Fatiha and another Surah",
        "Complete Ruku and Sujood",
        "Sit for final Tashahhud",
        "Recite Durood Ibrahim",
        "Recite closing duas",
        "Complete with Salam to both sides"
      ],
      image: images.tashahhud
    }
  ];

  // Animation handlers
  const handleNext = () => {
    if (currentStep < fajrSteps.length - 1) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <View style={prayerStyles.container}>
      <View style={prayerStyles.navigation}>
        <TouchableOpacity 
          onPress={handlePrevious}
          disabled={currentStep === 0}
          style={[prayerStyles.navButton, currentStep === 0 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-back" size={24} color="#16a34a" />
        </TouchableOpacity>
        
        <Text style={prayerStyles.stepText}>
          Step {currentStep + 1} of {fajrSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === fajrSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === fajrSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {fajrSteps[currentStep].title}
          </Text>
          
          <Image 
            source={fajrSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {fajrSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {fajrSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {fajrSteps.map((_, index) => (
          <View
            key={index}
            style={[
              prayerStyles.progressDot,
              index === currentStep && prayerStyles.activeDot
            ]}
          />
        ))}
      </View>
    </View>
  );
}
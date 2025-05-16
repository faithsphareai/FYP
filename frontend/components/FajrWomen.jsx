import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function FajrWomen() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  const fajrSteps = [
    {
      title: "Preparation",
      description: "Prepare for Fajr prayer with proper covering",
      details: [
        "Ensure proper covering - loose clothes covering entire body except face and hands",
        "Make sure prayer area is clean",
        "Face the Qibla direction",
        "Make wudu if not already done",
        "Keep movements gentle and controlled",
        "Ensure it's after dawn but before sunrise"
      ],
      image: images.niyyah
    },
    {
      title: "Sunnah First Rakat",
      description: "Begin with 2 Sunnah rakats",
      details: [
        "Make Niyyah for Sunnah prayer",
        "Say Takbir keeping hands below shoulders",
        "Place hands on chest, right over left",
        "Keep feet together and body compact",
        "Recite Thana quietly",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah"
      ],
      image: images.qiyam
    },
    {
      title: "Ruku and Sujood",
      description: "Complete first Sunnah rakat",
      details: [
        "Bend for Ruku keeping body compact",
        "Keep arms close to sides",
        "Say 'Subhana Rabbiyal Azeem' 3 times",
        "Rise saying 'Sami Allahu Liman Hamidah'",
        "For Sujood, go down keeping body compact",
        "Keep arms close to body in Sujood",
        "Maintain modest positioning"
      ],
      image: images.ruku
    },
    {
      title: "Second Sunnah Rakat",
      description: "Complete Sunnah prayer",
      details: [
        "Stand gracefully for second Rakat",
        "Recite Al-Fatiha and another Surah",
        "Complete Ruku and Sujood modestly",
        "Sit for final Tashahhud",
        "Keep feet to right side while sitting",
        "Recite At-Tahiyyat",
        "Complete with Salam"
      ],
      image: images.tashahhud
    },
    {
      title: "Farz First Rakat",
      description: "Begin 2 Farz rakats",
      details: [
        "Make new Niyyah for Farz",
        "Keep movements controlled",
        "Maintain proper covering",
        "Recite Al-Fatiha and another Surah",
        "Complete Ruku and Sujood",
        "Keep posture modest throughout"
      ],
      image: images.takbeer
    },
    {
      title: "Farz Completion",
      description: "Complete Fajr Farz prayer",
      details: [
        "Stand for second Rakat",
        "Recite required Surahs",
        "Complete positions modestly",
        "Sit for final Tashahhud",
        "Recite Durood and duas",
        "Complete with Salam",
        "Make personal duas"
      ],
      image: images.tasleem
    }
  ];

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
import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function Isha() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  const ishaSteps = [
    {
      title: "Initial Sunnah",
      description: "Begin with 4 Sunnah rakats of Isha",
      details: [
        "Make Niyyah for 4 Sunnah rakats",
        "Say Takbir-e-Tahreema",
        "Recite Thana",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood",
        "After 2 rakats, complete with Tashahhud",
        "Stand for remaining 2 rakats"
      ],
      image: images.niyyah
    },
    {
      title: "Farz First Rakat",
      description: "Begin 4 Farz rakats of Isha",
      details: [
        "Make new Niyyah for Isha Farz",
        "Say Takbir-e-Tahreema",
        "Recite Thana",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood"
      ],
      image: images.takbeer
    },
    {
      title: "Farz Second Rakat",
      description: "Complete second Farz rakat",
      details: [
        "Stand for second Rakat",
        "Recite Bismillah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood",
        "Sit for Qaida-e-Oola",
        "Recite At-Tahiyyat",
        "Stand for third Rakat"
      ],
      image: images.ruku
    },
    {
      title: "Farz Third and Fourth Rakat",
      description: "Complete remaining Farz rakats",
      details: [
        "In third and fourth rakats:",
        "Recite only Surah Al-Fatiha",
        "Complete Ruku and Sujood",
        "In final sitting recite:",
        "At-Tahiyyat",
        "Durood Ibrahim",
        "Closing duas",
        "Complete with Salam"
      ],
      image: images.sajda
    },
    {
      title: "Additional Sunnah",
      description: "Perform 2 Sunnah rakats",
      details: [
        "Make Niyyah for 2 Sunnah",
        "Complete 2 rakats with:",
        "Surah Al-Fatiha",
        "Another Surah in each rakat",
        "Ruku and Sujood",
        "Final sitting with At-Tahiyyat",
        "Complete with Salam"
      ],
      image: images.qiyam
    },
    {
      title: "Witr Prayer",
      description: "Complete 3 Witr rakats",
      details: [
        "Make Niyyah for Witr",
        "First two rakats like regular prayer",
        "In third rakat after Surah:",
        "Say Takbir and raise hands",
        "Recite Dua Qunoot",
        "Complete Ruku and Sujood",
        "End with Tashahhud and Salam"
      ],
      image: images.tashahhud
    }
  ];

  const handleNext = () => {
    if (currentStep < ishaSteps.length - 1) {
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
          Step {currentStep + 1} of {ishaSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === ishaSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === ishaSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {ishaSteps[currentStep].title}
          </Text>
          
          <Image 
            source={ishaSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {ishaSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {ishaSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {ishaSteps.map((_, index) => (
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
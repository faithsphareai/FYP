import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function Maghrib() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  const maghribSteps = [
    {
      title: "Initial Sunnah",
      description: "Begin with 2 Sunnah rakats of Maghrib",
      details: [
        "Make Niyyah for Sunnah prayer",
        "Say Takbir-e-Tahreema",
        "Recite Thana",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood",
        "Complete with Tashahhud and Salam"
      ],
      image: images.niyyah
    },
    {
      title: "Farz First Rakat",
      description: "Begin 3 Farz rakats of Maghrib",
      details: [
        "Make new Niyyah for Maghrib Farz",
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
      title: "Farz Third Rakat",
      description: "Complete final Farz rakat",
      details: [
        "Recite only Surah Al-Fatiha",
        "Complete Ruku and Sujood",
        "Sit for final Qaida",
        "Recite At-Tahiyyat",
        "Recite Durood Ibrahim",
        "Recite closing duas",
        "Complete with Salam"
      ],
      image: images.sajda
    },
    {
      title: "Additional Sunnah",
      description: "Complete 2 Sunnah rakats",
      details: [
        "Make Niyyah for 2 Sunnah",
        "Complete 2 rakats with:",
        "Surah Al-Fatiha",
        "Another Surah in each rakat",
        "Ruku and Sujood",
        "Final sitting with At-Tahiyyat",
        "Complete with Salam"
      ],
      image: images.tashahhud
    },
    {
      title: "Nafl Prayer",
      description: "Optional 2 Nafl rakats",
      details: [
        "Make Niyyah for Nafl",
        "Complete 2 rakats as normal",
        "Recite preferred Surahs",
        "Maintain focus and concentration",
        "Complete with Tashahhud",
        "End with Salam",
        "Make personal duas"
      ],
      image: images.tasleem
    }
  ];

  const handleNext = () => {
    if (currentStep < maghribSteps.length - 1) {
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
          Step {currentStep + 1} of {maghribSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === maghribSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === maghribSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {maghribSteps[currentStep].title}
          </Text>
          
          <Image 
            source={maghribSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {maghribSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {maghribSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {maghribSteps.map((_, index) => (
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
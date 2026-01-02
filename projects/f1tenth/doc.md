---
title: "26th Roboracer Autonomous Racing Competition at Techfest, IIT Bombay"
date: "2025-12-27"
excerpt: "2nd Place at the 26th Roboracer Autonomous Racing Competition, Techfest IIT Bombay."
coverImage: "./assets/car.png"
---

# 26th Roboracer Autonomous Racing Competition at Techfest, IIT Bombay

**Event**: 26th Roboracer Autonomous Racing Competition, Techfest IIT Bombay

From December 21st to 24th, 2025, my team—Team PulpFriction from BITS Pilani, Hyderabad Campus—took to the track at the 26th Roboracer Autonomous Racing Competition held at Techfest, IIT Bombay.

This event marked a significant milestone: it was the first time this competition was held in India. The second we caught a whiff of the announcement five months ago, we knew we had to be there. We immediately got to work. We got our car from performing wall following to forming a racing line, following it, overtaking, and more.

In the end, we secured 2nd place in the competition overall. This is how the competition unfolded.

## The Setup
We got our first look at the track on December 22nd. Visually, the map appeared simple, it's width was an issue. The track was narrow—ranging only 1.2 to 1.5 meters in width—leaving very little margin for error during overtakes.

However, the biggest adversary wasn't the geometry; it was the environment. The venue wasn't fully enclosed, meaning dust continuously settled on the track throughout the day. For an autonomous stack, this is a nightmare. It made System Identification (SysID) incredibly difficult, as the friction coefficients were constantly changing. We didn't have the luxury of windows to redo our SysID repeatedly, so we had to make a strategic pivot: rather than prioritizing raw speed and aggressive lap times, we had to tune our controller for reliability and robustness against variable traction.

## The Time Trials
December 23rd was Time Trials. Our strategy to balance speed with control paid off immediately. We went first, completing 32 laps in 5 minutes.

We clocked our fastest lap at 9.2 seconds, securing the 1st place position in the time trial round. Confidence was high.

![Time-Trials](assets/race.mp4)

## The Finals
December 24th brought the Head-to-Head rounds. We carried our momentum forward, winning our first two races cleanly to qualify for the Grand Finals.

However, the schedule played a massive role in the final outcome. There was a significant time gap between our semi-final qualification and the final race. During those hours, the non-indoor nature of the venue took its toll—a thick layer of dust had settled on the track.

The critical issue was that our controller tuning and dynamics parameters were set based on track conditions from early that morning. When the finals began, the car behaved noticeably differently on the slipperier surface. This was a big issue; all it took was one slip, leading to a crash. That error put us a couple of seconds behind, and despite trying our best to recover, we finished in 2nd Place.


The ending was a roller coaster of emotions. To dominate the time trials and reach the finals, only to be undone by environmental factors, was tough. However, looking back at the five-month sprint from simple wall-following to a podium finish at an event of this scale, the experience was incredible.
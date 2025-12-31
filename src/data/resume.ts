export interface SocialLink {
    platform: string;
    url: string;
    label?: string;
}

export interface Education {
    school: string;
    degree: string;
    year: string;
    location: string;
}

export interface SkillCategory {
    category: string;
    skills: string[];
}

export interface Position {
    role: string;
    organization: string;
    period: string;
    description: string[];
}

export const resumeData = {
    name: "Arjun Murali",
    email: "arjunmurali215@gmail.com",
    phone: "+91 9148494291",
    socials: [
        {
            platform: "LinkedIn",
            url: "https://linkedin.com/in/arjunmurali215",
            label: "linkedin.com/in/arjunmurali215"
        },
        {
            platform: "GitHub",
            url: "https://github.com/arjunmurali215",
            label: "github.com/arjunmurali215"
        }
    ] as SocialLink[],
    education: [
        {
            school: "BITS Pilani, Hyderabad Campus",
            degree: "B.E. Electronics and Instrumentation (Third Year)",
            year: "Aug 2023 – May 2027",
            location: "Hyderabad, India"
        }
    ] as Education[],
    skills: [
        {
            category: "Languages",
            skills: ["Python", "C++", "Java", "Dart (Flutter)"]
        },
        {
            category: "Frameworks",
            skills: ["PyTorch", "ROS 1 & 2", "Gazebo"]
        },
        {
            category: "Design Tools",
            skills: ["SolidWorks", "Fusion 360"]
        },
        {
            category: "Concepts",
            skills: ["Manipulators", "Computer Vision", "Kinematics", "SLAM"]
        }
    ] as SkillCategory[],
    leadership: [
        {
            role: "President",
            organization: "Automation and Robotics Club, BITS Hyderabad",
            period: "Apr 2025 – Present",
            description: [
                "Spearheading educational initiatives by organizing hands-on workshops on computer vision, CAD design, 3D printing, and microcontrollers, engaging over 150 students.",
                "Serving as the official representative and primary liaison between the club, faculty, and external organizations.",
                "Leading a 60-member core team to plan and execute flagship projects, inter-college competitions, and long-term technical goals."
            ]
        }
    ] as Position[]
};

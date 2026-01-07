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
            degree: "B.E. Electronics and Instrumentation (CGPA: 8.23/10)\nMinor in Robotics and Automation (GPA: 9/10)",
            year: "Aug 2023 – May 2027",
            location: "Hyderabad, India"
        }
    ] as Education[],
    skills: [
        {
            category: "Languages",
            skills: ["Python", "C++", "Java", "Flutter"]
        },
        {
            category: "Frameworks",
            skills: ["PyTorch", "OpenCV", "ROS 1 & 2", "Gazebo", "MuJoCo"]
        },
        {
            category: "Design Tools",
            skills: ["SolidWorks", "Fusion 360", "KiCAD"]
        },
        {
            category: "Areas of Interest",
            skills: ["Manipulators", "Computer Vision", "Kinematics"]
        }
    ] as SkillCategory[],
    leadership: [
        {
            role: "President",
            organization: "Automation and Robotics Club, BITS Hyderabad",
            period: "Apr 2025 – Present",
            description: [
                "Leading one of the largest technical clubs on campus, managing a team of 60+ active members and mentoring junior projects.",
                "Spearheaded educational initiatives by organizing hands-on workshops on Computer Vision, CAD, and Embedded Systems, impacting over 150 students.",
                "Liaised with faculty and external organizations to secure funding and resources for student-led research initiatives."
            ]
        }
    ] as Position[]
};

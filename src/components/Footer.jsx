import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Github, Linkedin, Mail } from 'lucide-react';

const Footer = () => {
    const navigate = useNavigate();

    const scrollToSection = (sectionId) => {
        if (window.location.pathname.replace(/\/Portfolio/, '') === '/') {
            document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
        } else {
            navigate('/');
            setTimeout(() => {
                document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
            }, 150);
        }
    };

    return (
        <footer className="bg-bg-warm border-t border-rule py-12 sm:py-14 mt-auto">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center gap-5">
                <div>
                    <p className="font-display text-xl font-semibold text-ink mb-1">Saurav Kumar</p>
                    <p className="text-[13px] text-ink-muted">ML Engineer · Generative AI</p>
                </div>

                <div className="flex justify-center gap-8">
                    <button onClick={() => scrollToSection('projects')} className="text-[13px] text-ink-muted hover:text-ink transition-colors bg-transparent border-none cursor-pointer">
                        Projects
                    </button>
                    <button onClick={() => scrollToSection('resume')} className="text-[13px] text-ink-muted hover:text-ink transition-colors bg-transparent border-none cursor-pointer">
                        Resume
                    </button>
                    <Link to="/blogs" className="text-[13px] text-ink-muted hover:text-ink transition-colors">
                        Writing
                    </Link>
                </div>

                <div className="flex justify-center gap-5 pt-1">
                    <a href="https://github.com/SK-15" target="_blank" rel="noopener noreferrer" className="text-ink-faint hover:text-accent transition-colors duration-200" aria-label="GitHub">
                        <Github size={18} />
                    </a>
                    <a href="https://www.linkedin.com/in/sauravkumar1503/" target="_blank" rel="noopener noreferrer" className="text-ink-faint hover:text-accent transition-colors duration-200" aria-label="LinkedIn">
                        <Linkedin size={18} />
                    </a>
                    <a href="mailto:sauravkumar585@gmail.com" className="text-ink-faint hover:text-accent transition-colors duration-200" aria-label="Email">
                        <Mail size={18} />
                    </a>
                </div>

                <p className="text-[13px] text-ink-muted pt-2">
                    © {new Date().getFullYear()} Saurav Kumar. Built with React &amp; Tailwind.
                </p>
            </div>
        </footer>
    );
};

export default Footer;

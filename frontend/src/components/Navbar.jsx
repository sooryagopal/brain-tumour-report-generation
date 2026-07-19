import { NavLink, useNavigate } from 'react-router-dom';
import { Brain, Activity, Info, Scan } from 'lucide-react';

const navItems = [
  { label: 'Home',    to: '/',        icon: Brain },
  { label: 'Analyze', to: '/analyze', icon: Scan },
  { label: 'About',   to: '/about',   icon: Info },
];

export default function Navbar() {
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Logo */}
        <NavLink to="/" className="nav-logo" style={{ textDecoration: 'none' }}>
          <div className="nav-logo-icon">🧠</div>
          <span className="nav-logo-text">
            Neuro<span className="gradient-text">Scan AI</span>
          </span>
        </NavLink>

        {/* Navigation Links */}
        <ul className="nav-links">
          {navItems.map(({ label, to, icon: Icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) =>
                  `nav-link${isActive ? ' active' : ''}`
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        {/* CTA Button */}
        <button
          className="btn-primary"
          style={{ padding: '8px 20px', fontSize: '0.85rem' }}
          onClick={() => navigate('/analyze')}
        >
          <Activity size={15} />
          Analyse MRI
        </button>
      </div>
    </nav>
  );
}

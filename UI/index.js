import React from "react";
import { Nav, NavLink, NavMenu }
	from "./NavbarElements";

const Navbar = () => {
return (
	<>
	<Nav>
		<NavMenu>
		<NavLink to="/dashboard" activeStyle>
			Dashboard
		</NavLink>
		<NavLink to="/upload" activeStyle>
			Upload Image
		</NavLink>
		</NavMenu>
	</Nav>
	</>
);
};

export default Navbar;

#!/usr/bin/env python3
"""
Quick test of REFPROP oxygen properties to verify installation and functionality
"""

from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import os
import numpy as np

def test_oxygen_properties():
    """Test basic oxygen property calculations"""
    try:
        # Set up REFPROP
        REFPROP_PATH = r"C:\Program Files (x86)\REFPROP"
        RP = REFPROPFunctionLibrary(os.path.join(REFPROP_PATH, "REFPRP64.dll"))
        RP.SETPATHdll(REFPROP_PATH)
        
        print("REFPROP Oxygen Properties Test")
        print("=" * 40)
        
        # Test conditions - typical LOX conditions
        T = 90.0    # Temperature [K] 
        P = 3.5e6   # Pressure [Pa] = 3.5 MPa
        z = [1.0]   # Pure oxygen
        
        # Get basic properties
        result = RP.REFPROPdll("OXYGEN", "TP", "D;H;S;VIS;TCX;CP", RP.MASS_BASE_SI, 0, 0, T, P, z)
        
        if result.ierr == 0:
            density = result.Output[0]              # kg/m³
            enthalpy = result.Output[1]             # J/kg
            entropy = result.Output[2]              # J/kg·K
            viscosity = result.Output[3]            # Pa·s
            thermal_conductivity = result.Output[4] # W/m·K
            specific_heat = result.Output[5]        # J/kg·K
            
            prandtl = (viscosity * specific_heat) / thermal_conductivity
            
            print(f"Temperature: {T} K ({T-273.15:.1f} °C)")
            print(f"Pressure: {P/1e6:.2f} MPa")
            print("-" * 30)
            print(f"Density: {density:.2f} kg/m³")
            print(f"Viscosity: {viscosity*1e6:.2f} μPa·s")
            print(f"Thermal Conductivity: {thermal_conductivity:.4f} W/m·K")
            print(f"Specific Heat: {specific_heat/1000:.3f} kJ/kg·K")
            print(f"Prandtl Number: {prandtl:.4f}")
            
            # Test property range
            print("\n" + "=" * 40)
            print("Testing property ranges:")
            
            # Temperature range at constant pressure
            T_range = [80, 85, 90, 95, 100, 110, 120]
            print(f"\nAt P = {P/1e6:.1f} MPa:")
            print("T [K]   Density [kg/m³]   Viscosity [μPa·s]")
            print("-" * 45)
            
            for T_test in T_range:
                try:
                    result = RP.REFPROPdll("OXYGEN", "TP", "D;VIS", RP.MASS_BASE_SI, 0, 0, T_test, P, z)
                    if result.ierr == 0:
                        rho = result.Output[0]
                        visc = result.Output[1]
                        print(f"{T_test:4.0f}    {rho:8.2f}        {visc*1e6:8.2f}")
                    else:
                        print(f"{T_test:4.0f}    Error: {result.herr}")
                except:
                    print(f"{T_test:4.0f}    Calculation failed")
            
            print("\n✅ REFPROP oxygen properties test PASSED")
            return True
            
        else:
            print(f"❌ REFPROP Error: {result.herr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\nPossible issues:")
        print("1. REFPROP not installed or path incorrect")
        print("2. ctREFPROP Python package not installed")
        print("3. Oxygen fluid file not found in REFPROP")
        return False

if __name__ == "__main__":
    success = test_oxygen_properties()
    if success:
        print("\n🎉 Ready to use REFPROP for oxygen cooling analysis!")
    else:
        print("\n⚠️  Please check REFPROP installation and try again.")
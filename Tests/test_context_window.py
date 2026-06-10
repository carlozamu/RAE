import asyncio
import random
import re
from Utils.LLM import LLM 

async def run_reasoning_depth_test(mode="2D"):
    llm = LLM(model_name="google/gemma-3-1b-it", base_url="http://localhost:8000") 
    
    depths_to_test = [2, 4, 5, 7, 10, 15]
    trials_per_depth = 20 
    
    summary_results = []
    
    print(f"🚀 Starting Cognitive Event Horizon Test | MODE: {mode.upper()}")
    print(f"Testing depths: {depths_to_test} with {trials_per_depth} trials each.\n")

    for depth in depths_to_test:
        success_count = 0
        total_tokens_for_depth = 0
        print(f"--- Running Tests for Depth: {depth} ---")
        
        for trial in range(trials_per_depth):
            prompt = ""
            
            # --- CONFIGURE MODE SPECIFICS ---
            if mode == "1D":
                true_pos = 0
                moves = [("Up", 1), ("Down", -1)]
                setup_rules = (
                    "We are tracking an elevator in a shaft. It starts at floor 0.\n"
                    "Here are the rules:\n"
                    "- Moving Up adds 1 to the floor number. (last position +1)\n"
                    "- Moving Down subtracts 1 from the floor number. (last position -1)\n"
                    "Acknowledge these rules and confirm the start by replying exactly with: 'Starting Position: 0'.\n"
                )
            elif mode == "2D":
                true_x, true_y = 0, 0
                moves = [
                    ("North", 0, 1), 
                    ("South", 0, -1),
                    ("East", 1, 0), 
                    ("West", -1, 0)
                ]
                setup_rules = (
                    "We are tracking a robot on a 2D grid. The robot starts at position (0, 0).\n"
                    "Here are the rules for movement:\n"
                    "- Moving North adds 1 to the Y coordinate.\n"
                    "- Moving South subtracts 1 from the Y coordinate.\n"
                    "- Moving East adds 1 to the X coordinate.\n"
                    "- Moving West subtracts 1 from the X coordinate.\n"
                    "Acknowledge these rules and confirm the starting position by replying exactly with: 'Starting Position: (0, 0)'.\n"
                )
            elif mode == "3D":
                true_x, true_y, true_z = 0, 0, 0
                moves = [
                    ("North", 0, 1, 0), ("South", 0, -1, 0),
                    ("East", 1, 0, 0), ("West", -1, 0, 0),
                    ("Up", 0, 0, 1), ("Down", 0, 0, -1)
                ]
                setup_rules = (
                    "We are tracking a drone in 3D space. It starts at (0, 0, 0).\n"
                    "Here are the rules:\n"
                    "- North adds 1 to Y. South subtracts 1 from Y.\n"
                    "- East adds 1 to X. West subtracts 1 from X.\n"
                    "- Up adds 1 to Z. Down subtracts 1 from Z.\n"
                    "Acknowledge these rules and confirm the start by replying exactly with: 'Starting Position: (0, 0, 0)'.\n"
                )
            
            # --- 1. SETUP NODE ---
            prompt += f"<start_of_turn>user\n{setup_rules}<end_of_turn>\n<start_of_turn>model\n"
            setup_response = await llm.generate_text(user_prompt=prompt, temperature=0.0)
            prompt += setup_response.strip() + "\n<end_of_turn>\n"
            
            # --- 2. INTERMEDIATE NODES ---
            for i in range(depth):
                if mode == "1D":
                    direction, val = random.choice(moves)
                    true_pos += val
                    format_instruction = "'Last Position: F' (where F is the floor number)"
                elif mode == "2D":
                    direction, dx, dy = random.choice(moves)
                    true_x += dx; true_y += dy
                    format_instruction = "'Last Position: (X, Y)'"
                elif mode == "3D":
                    direction, dx, dy, dz = random.choice(moves)
                    true_x += dx; true_y += dy; true_z += dz
                    format_instruction = "'Last Position: (X, Y, Z)'"

                prompt += "<start_of_turn>user\n"
                prompt += f"Step {i+1}: Move 1 unit {direction}. Apply the rule for {direction} to the last position computed to get the updated position. Format your answer exactly as: {format_instruction}.\n"
                prompt += "<end_of_turn>\n<start_of_turn>model\n"
                
                inter_response = await llm.generate_text(user_prompt=prompt, temperature=0.0)
                prompt += inter_response.strip() + "\n<end_of_turn>\n"

            # --- 3. END NODE ---
            prompt += "<start_of_turn>user\n"
            if mode == "1D":
                prompt += "Final Task: What is the current final floor? Output ONLY the number of the exact current floor, with no other text.\n"
            elif mode == "2D":
                prompt += "Final Task: What is the current final (X, Y) coordinate of the robot? Output ONLY the coordinate in the format (X, Y) with no other text.\n"
            elif mode == "3D":
                prompt += "Final Task: What is the final (X, Y, Z) coordinate? Output ONLY the coordinate in the format (X, Y, Z) with no other text.\n"
            prompt += "<end_of_turn>\n<start_of_turn>model\n"

            final_response = await llm.generate_text(user_prompt=prompt, temperature=0.0, max_tokens=15)
            last_output = final_response.strip()
            
            # --- 4. EVALUATION ---
            # --- 4. EVALUATION (Robust Regex Parsing) ---
            if mode == "1D":
                expected_coord = str(true_pos)
                # Find all numbers (including negatives) in the response text
                numbers_found = re.findall(r'-?\d+', last_output)
                
                # Check if the model's final conclusion matches the expected floor.
                # Taking the last number [-1] prevents false positives if the model 
                # rambles about previous steps before giving the final answer.
                is_correct = bool(numbers_found) and numbers_found[-1] == expected_coord
                
            elif mode == "2D":
                expected_coord = f"({true_x}, {true_y})"
                # Matches (X, Y) anywhere in the text, even if the model messes up the spaces 
                # Example matches: "(0, 1)", "(0,1)", "The answer is ( 0 , 1 )."
                pattern = rf"\(\s*{true_x}\s*,\s*{true_y}\s*\)"
                is_correct = bool(re.search(pattern, last_output))
                
            elif mode == "3D":
                expected_coord = f"({true_x}, {true_y}, {true_z})"
                # Matches (X, Y, Z) with flexible spacing
                pattern = rf"\(\s*{true_x}\s*,\s*{true_y}\s*,\s*{true_z}\s*\)"
                is_correct = bool(re.search(pattern, last_output))
            
            if is_correct:
                success_count += 1
                
            total_text = prompt + last_output
            #print(f"Trial {trial+1} Prompt and Response:\n{total_text}\n{'-'*50}")
            approx_tokens = int(len(total_text.split()) / 4)  # Rough estimate: 4 characters per token
            total_tokens_for_depth += approx_tokens

            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"  Trial {trial+1:02d}/{trials_per_depth} | Expected: {expected_coord:<10} | Got: {last_output[:25]:<35} | {status}")

        # --- 5. DEPTH AVERAGES ---
        success_rate = (success_count / trials_per_depth) * 100
        avg_tokens = int(total_tokens_for_depth / trials_per_depth)
        
        summary_results.append({"depth": depth, "avg_tokens": avg_tokens, "success_rate": success_rate})
        print(f"➡️ Depth {depth} Completed. Average Success: {success_rate:.1f}%\n")

    # --- 6. FINAL REPORT RECAP ---
    print("\n" + "="*65)
    print(f"🏆 FINAL COGNITIVE EVENT HORIZON REPORT | {mode.upper()} MODE")
    print("="*65)
    print(f"{'Graph Depth':<15} | {'Avg Context Tokens':<20} | {'Success Rate'}")
    print("-" * 65)
    
    for res in summary_results:
        print(f"{res['depth']:<15} | {res['avg_tokens']:<20} | {res['success_rate']:>7.1f}%")
        
    print("="*65)

if __name__ == "__main__":
    # Toggle between "1D", "2D", or "3D" to run the specific test profile
    asyncio.run(run_reasoning_depth_test(mode="3D"))
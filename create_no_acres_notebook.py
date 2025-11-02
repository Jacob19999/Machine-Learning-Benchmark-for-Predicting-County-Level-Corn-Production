import json

print("Reading Model_Comparison.ipynb...")
with open('Model_Comparison.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

# Find the data loading cell (cell with feature_cols definition)
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    # Update the title cell (cell 0)
    if 'Corn Yield Prediction - Model Comparison (Phase 3)' in source and cell['cell_type'] == 'markdown':
        new_title = source.replace(
            'Corn Yield Prediction - Model Comparison (Phase 3)',
            'Corn Yield Prediction - Model Comparison (Phase 3) - WITHOUT corn_acres_planted'
        )
        new_title = new_title.replace(
            '- **Features:** 66 engineered features',
            '- **Features:** 65 engineered features (excluding corn_acres_planted)'
        )
        nb['cells'][i]['source'] = new_title.split('\n')
        print(f"Updated title cell at index {i}")
    
    # Find the data loading cell with feature selection
    if 'feature_cols = [' in source and 'ID_COLS' in source:
        print(f"Found feature selection cell at index {i}")
        
        # Modify the feature selection to exclude corn_acres_planted
        original_code = ''.join(cell['source'])
        
        # Add exclusion for corn_acres_planted
        new_code = original_code
        
        # Find the line that defines feature_cols
        lines = new_code.split('\n')
        for j, line in enumerate(lines):
            if 'feature_cols = [' in line or ('feature_cols' in line and 'if col not in' in line):
                # After feature_cols is defined, add exclusion
                # Find where feature_cols assignment ends
                if j + 1 < len(lines):
                    # Insert exclusion after feature_cols definition
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    exclusion_line = ' ' * indent + "# Exclude corn_acres_planted from features\n"
                    exclusion_line += ' ' * indent + "feature_cols = [col for col in feature_cols if col != 'corn_acres_planted']\n"
                    
                    # Find the right place to insert (after feature_cols definition is complete)
                    insert_idx = j + 1
                    # Skip continuation lines
                    while insert_idx < len(lines) and (lines[insert_idx].strip().startswith(']') or 
                                                       'if col not in' in lines[insert_idx] or
                                                       'and df[col].dtype' in lines[insert_idx]):
                        insert_idx += 1
                    
                    lines.insert(insert_idx, exclusion_line)
                    break
        
        # Alternative approach: modify the feature_cols line directly
        if 'feature_cols = [col for col in df.columns' in new_code:
            # Replace the feature_cols definition
            new_code = new_code.replace(
                "feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL]]",
                "feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL] and col != 'corn_acres_planted']"
            )
        elif 'feature_cols = [' in new_code:
            # More complex case - add after the list comprehension
            new_code = new_code.replace(
                "feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL]]",
                "feature_cols = [col for col in df.columns if col not in ID_COLS + [TARGET_COL] and col != 'corn_acres_planted']"
            )
        
        # If not found, add exclusion line after feature_cols definition
        if "col != 'corn_acres_planted'" not in new_code:
            # Add exclusion after feature_cols definition
            if 'feature_cols = [' in new_code:
                # Find the end of feature_cols definition
                lines = new_code.split('\n')
                for idx, line in enumerate(lines):
                    if 'feature_cols = [' in line or 'feature_cols = [' in line:
                        # Find the end of this definition (next line that's not indented or empty)
                        end_idx = idx + 1
                        while end_idx < len(lines) and (lines[end_idx].strip() == '' or 
                                                       lines[end_idx].startswith(' ') or
                                                       lines[end_idx].strip().startswith('#')):
                            end_idx += 1
                        
                        # Insert exclusion
                        indent = len(line) - len(line.lstrip())
                        exclusion = f"{' ' * indent}# Exclude corn_acres_planted from features\n"
                        exclusion += f"{' ' * indent}feature_cols = [col for col in feature_cols if col != 'corn_acres_planted']\n"
                        lines.insert(end_idx, exclusion)
                        new_code = '\n'.join(lines)
                        break
        
        nb['cells'][i]['source'] = new_code.split('\n')
        print(f"Updated feature selection cell at index {i}")

# Save the new notebook
output_file = 'Model_Comparison_NoAcres.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Created new notebook: {output_file}")
print("   - Excluded 'corn_acres_planted' from feature set")
print("   - All other models and code remain identical")


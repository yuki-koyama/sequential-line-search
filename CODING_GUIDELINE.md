# Coding Guideline

The current source codes in this repository are really messy; they follow many inconsistent coding styles and naming convention. This guideline is added for easing the current painful situation as much as possible in the future development.

## Naming Convention

### Class

`MyClass`

### Class member attribute

`m_member_attribute`

- This rule is applied for both public and private member attributes.

### Class member method

`CalculateValue`

- The name should begin with a verb, following an object if appropriate.
- The `CamelCase` rule is applied even when mentioning mathematical variables.

### Struct

`MyStruct`

### Struct member attribute

`member_attribute`

### Struct member method

`(N/A)`

- Never define methods for structs; use classes or functions instead.

### Function

`CalculateValue`

- The name should begin with a verb, following an object if appropriate.
- The `CamelCase` rule is applied even when mentioning mathematical variables.

### Variable

`local_variable`

### Mathematical variable

`x_i`, `K_f`, `C_inv`, etc.

- It is fine to use capitals if appropriate although this breaks the `snake_case` rule for other variables.

### Boolean variable

`has_something`, `is_valid`, etc.

### Lambda

`calculate_value`, `f`, etc.

### Namespace

`my_library_namespace`

## Abbreviations

- `num`: the number of
- `dim`: the number of dimensions
- `inv`: inverse
- `grad`: gradient
- `calc`: calculate
- `iter`: iteration
- `dir`: directory
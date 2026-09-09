# frozen_string_literal: true

require 'json'

module AllocationSites
  class Invalid < StandardError; end

  class UniqueObject < Hash
    def []=(key, value)
      raise Invalid, "duplicate JSON key #{key.inspect}" if key?(key)
      super
    end
  end

  S64_MIN = -(1 << 63)
  S64_MAX = (1 << 63) - 1
  U64_MAX = (1 << 64) - 1
  U32_MAX = (1 << 32) - 1

  PROFILE_FIELDS = %w[alloc_bytes free_bytes alloc_objects free_objects].freeze
  COUNTER_FIELDS = %w[total_alloc mallocs frees heap_alloc heap_objects num_gc].freeze
  BOUNDARY_NAMES = %w[pre_raw_before start end post_gc post_raw].freeze
  CAPTURE_KEYS = %w[
    schema control n warmup_calls completed_calls go_version goos goarch
    gomaxprocs godebug gogc gomemlimit profile_rate_before profile_rate_after
    raw_capacity caller_function worker_function boundaries pre post region_error
    errors
  ].freeze
  TAIL_KEYS = %w[schema tail report_write_excluded].freeze
  SNAPSHOT_KEYS = %w[reported_count ok rows].freeze
  ROW_KEYS = %w[ordinal alloc_bytes free_bytes alloc_objects free_objects pcs frames].freeze
  FRAME_KEYS = %w[pc function file line].freeze

  EXPECTED_SCHEMA = 'goai-control-alloc-sites-v1'
  EXPECTED_TAIL_SCHEMA = 'goai-control-alloc-sites-tail-v1'
  DERIVED_SCHEMA = 'goai-control-alloc-sites-derived-v1'
  CONTROL = 'SoftplusF64_256K'
  CALLER = 'github.com/jxsl13/goai/backend/cpu_test.allocationSiteExecuteRegion'
  WORKER = 'github.com/jxsl13/goai/backend/cpu.poolWorker'
  INTERPRETATION = 'perturbed process-wide diagnostic; worker association only; no qualification'

  module_function

  def parse(text)
    unless text.is_a?(String)
      raise Invalid, '$: JSON input must be a string'
    end

    parsed = JSON.parse(text, object_class: UniqueObject, create_additions: false, allow_nan: false)
    reject_nonfinite!(parsed, '$')
    parsed
  rescue JSON::ParserError => e
    raise Invalid, "$: invalid JSON: #{e.message}"
  end

  def reject_nonfinite!(value, path)
    case value
    when Float
      invalid(path, 'nonfinite JSON number') unless value.finite?
    when Hash
      value.each { |key, child| reject_nonfinite!(child, "#{path}[#{key.inspect}]") }
    when Array
      value.each_with_index { |child, index| reject_nonfinite!(child, "#{path}[#{index}]") }
    end
    value
  end

  def analyze(capture, tail)
    object!(capture, '$capture', CAPTURE_KEYS)
    object!(tail, '$tail', TAIL_KEYS)

    literal!(capture['schema'], EXPECTED_SCHEMA, '$capture.schema')
    literal!(capture['control'], CONTROL, '$capture.control')
    literal!(capture['n'], 1024, '$capture.n')
    literal!(capture['warmup_calls'], 1, '$capture.warmup_calls')
    literal!(capture['completed_calls'], 1024, '$capture.completed_calls')
    nonempty_string!(capture['go_version'], '$capture.go_version')
    literal!(capture['goos'], 'darwin', '$capture.goos')
    literal!(capture['goarch'], 'arm64', '$capture.goarch')
    integer!(capture['gomaxprocs'], '$capture.gomaxprocs', 0, S64_MAX)
    unless [1, 12].include?(capture['gomaxprocs'])
      invalid('$capture.gomaxprocs', 'must be 1 or 12')
    end
    literal!(capture['godebug'], 'memprofilerate=1', '$capture.godebug')
    literal!(capture['gogc'], '100', '$capture.gogc')
    literal!(capture['gomemlimit'], 'off', '$capture.gomemlimit')
    literal!(capture['profile_rate_before'], 1, '$capture.profile_rate_before')
    literal!(capture['profile_rate_after'], 1, '$capture.profile_rate_after')
    literal!(capture['raw_capacity'], 65_536, '$capture.raw_capacity')
    literal!(capture['caller_function'], CALLER, '$capture.caller_function')
    literal!(capture['worker_function'], WORKER, '$capture.worker_function')
    literal!(capture['region_error'], '', '$capture.region_error')
    array!(capture['errors'], '$capture.errors')
    invalid('$capture.errors', 'must be empty') unless capture['errors'].empty?

    literal!(tail['schema'], EXPECTED_TAIL_SCHEMA, '$tail.schema')
    literal!(tail['report_write_excluded'], true, '$tail.report_write_excluded')

    boundaries = validate_boundaries(capture['boundaries'], tail['tail'])
    pre_rows, pre_zero = validate_snapshot(capture['pre'], '$capture.pre')
    post_rows, post_zero = validate_snapshot(capture['post'], '$capture.post')

    raw_keys = build_raw_keys(pre_rows, post_rows, capture['caller_function'], capture['worker_function'])
    normalized_sites = build_normalized_sites(raw_keys)
    groups = build_groups(raw_keys, normalized_sites)
    profile_totals = totals_from_groups(groups)
    unless groups['caller']['delta']['alloc_bytes'] > 0 &&
           groups['caller']['delta']['alloc_objects'] > 0
      invalid('$capture', 'caller delta must contribute positive alloc_bytes and alloc_objects')
    end

    windows = build_windows(boundaries)
    immediate_minus_profile = {
      'alloc_bytes' => checked_sub(windows['region']['total_alloc'], profile_totals['delta']['alloc_bytes'], '$derived.immediate_minus_profile.alloc_bytes'),
      'alloc_objects' => checked_sub(windows['region']['mallocs'], profile_totals['delta']['alloc_objects'], '$derived.immediate_minus_profile.alloc_objects'),
      'free_objects' => checked_sub(windows['region']['frees'], profile_totals['delta']['free_objects'], '$derived.immediate_minus_profile.free_objects')
    }

    {
      'schema' => DERIVED_SCHEMA,
      'control' => CONTROL,
      'n' => 1024,
      'gomaxprocs' => capture['gomaxprocs'],
      'caller_function' => capture['caller_function'],
      'worker_function' => capture['worker_function'],
      'windows' => windows,
      'raw_keys' => raw_keys,
      'normalized_sites' => normalized_sites,
      'zero_active' => pre_zero + post_zero,
      'groups' => groups,
      'profile_totals' => profile_totals,
      'immediate_minus_profile' => immediate_minus_profile,
      'interpretation' => INTERPRETATION
    }
  end

  def validate_boundaries(value, tail_value)
    object!(value, '$capture.boundaries', BOUNDARY_NAMES)
    values = BOUNDARY_NAMES.map do |name|
      [name, validate_counters(value[name], "$capture.boundaries.#{name}")]
    end
    tail_counter = validate_counters(tail_value, '$tail.tail')
    ordered = values + [['tail', tail_counter]]

    %w[total_alloc mallocs frees num_gc].each do |field|
      ordered.each_cons(2) do |left, right|
        if right[1][field] < left[1][field]
          invalid("$capture.boundaries.#{right[0]}.#{field}", "decreases from #{left[0]}")
        end
      end
    end

    result = {}
    values.each { |name, counters| result[name] = counters }
    result['tail'] = tail_counter
    result
  end

  def validate_counters(value, path)
    object!(value, path, COUNTER_FIELDS)
    result = {}
    COUNTER_FIELDS.each do |field|
      max = field == 'num_gc' ? U32_MAX : U64_MAX
      result[field] = integer!(value[field], "#{path}.#{field}", 0, max)
    end
    result
  end

  def validate_snapshot(value, path)
    object!(value, path, SNAPSHOT_KEYS)
    count = integer!(value['reported_count'], "#{path}.reported_count", 0, S64_MAX)
    literal!(value['ok'], true, "#{path}.ok")
    array!(value['rows'], "#{path}.rows")
    invalid("#{path}.reported_count", 'exceeds raw capacity 65536') if count > 65_536
    unless value['rows'].length == count
      invalid("#{path}.rows", "length #{value['rows'].length} does not match reported_count #{count}")
    end

    active = []
    zero = []
    value['rows'].each_with_index do |row, ordinal|
      row_path = "#{path}.rows[#{ordinal}]"
      object!(row, row_path, ROW_KEYS)
      literal!(row['ordinal'], ordinal, "#{row_path}.ordinal")
      counters = {}
      PROFILE_FIELDS.each do |field|
        counters[field] = integer!(row[field], "#{row_path}.#{field}", 0, S64_MAX)
      end
      pcs, prefix = validate_pcs(row['pcs'], "#{row_path}.pcs")
      functions = validate_frames(row['frames'], "#{row_path}.frames", prefix, counters['alloc_objects'] > 0)

      if counters['alloc_objects'] == 0
        unless PROFILE_FIELDS.all? { |field| counters[field] == 0 }
          invalid(row_path, 'alloc_objects=0 requires all four profile counters to be zero')
        end
        zero << {'snapshot' => path.end_with?('.pre') ? 'pre' : 'post', 'ordinal' => ordinal}
        next
      end

      invalid("#{row_path}.alloc_bytes", 'must be positive for an active row') unless counters['alloc_bytes'] > 0
      if counters['free_objects'] > counters['alloc_objects']
        invalid("#{row_path}.free_objects", 'exceeds alloc_objects')
      end
      if counters['free_bytes'] > counters['alloc_bytes']
        invalid("#{row_path}.free_bytes", 'exceeds alloc_bytes')
      end
      unless (counters['alloc_bytes'] % counters['alloc_objects']).zero?
        invalid("#{row_path}.alloc_bytes", 'is not exactly divisible by alloc_objects')
      end
      slot_size = counters['alloc_bytes'] / counters['alloc_objects']
      integer!(slot_size, "#{row_path}.slot_size", 1, S64_MAX)
      product = checked_mul(counters['free_objects'], slot_size, "#{row_path}.free_bytes")
      unless counters['free_bytes'] == product
        invalid("#{row_path}.free_bytes", 'does not equal free_objects * slot_size')
      end
      invalid("#{row_path}.pcs", 'active row has an empty PC prefix') if prefix.empty?
      invalid("#{row_path}.frames", 'active row has no resolved frames') if functions.empty?

      active << {
        'ordinal' => ordinal,
        'counters' => counters,
        'slot_size' => slot_size,
        'pcs' => prefix,
        'functions' => functions
      }
    end
    [active, zero]
  end

  def validate_pcs(value, path)
    array!(value, path)
    invalid(path, 'must contain exactly 32 PCs') unless value.length == 32
    prefix = []
    saw_zero = false
    value.each_with_index do |pc, index|
      number = canonical_pc!(pc, "#{path}[#{index}]")
      if number.zero?
        saw_zero = true
      elsif saw_zero
        invalid("#{path}[#{index}]", 'nonzero PC appears after the first zero')
      else
        prefix << pc
      end
    end
    [value, prefix]
  end

  def validate_frames(value, path, pc_prefix, active)
    array!(value, path)
    invalid(path, 'must be empty when the PC prefix is empty') if pc_prefix.empty? && !value.empty?
    functions = []
    value.each_with_index do |frame, index|
      frame_path = "#{path}[#{index}]"
      object!(frame, frame_path, FRAME_KEYS)
      canonical_pc!(frame['pc'], "#{frame_path}.pc")
      string!(frame['function'], "#{frame_path}.function")
      string!(frame['file'], "#{frame_path}.file")
      integer!(frame['line'], "#{frame_path}.line", 0, nil)
      if active && frame['function'].empty?
        invalid("#{frame_path}.function", 'must be nonempty for an active row')
      end
      functions << frame['function'].dup
    end
    functions
  end

  def build_raw_keys(pre_rows, post_rows, caller, worker)
    snapshots = {'pre' => aggregate_snapshot(pre_rows, '$capture.pre'), 'post' => aggregate_snapshot(post_rows, '$capture.post')}
    keys = (snapshots['pre'].keys + snapshots['post'].keys).uniq.sort
    keys.each_with_index.map do |key, index|
      pre = snapshots['pre'][key]
      post = snapshots['post'][key]
      functions = symbolization_for(pre, post, index)
      pre_counters = pre ? pre['counters'] : zero_profile
      post_counters = post ? post['counters'] : zero_profile
      delta = profile_sub(post_counters, pre_counters, "$derived.raw_keys[#{index}].delta", true)
      pre_ordinals = pre ? pre['ordinals'].sort : []
      post_ordinals = post ? post['ordinals'].sort : []
      {
        'index' => index,
        'slot_size' => key[0],
        'pcs' => key[1].dup,
        'functions' => functions.dup,
        'group' => group_for(functions, caller, worker),
        'pre' => pre_counters.dup,
        'post' => post_counters.dup,
        'delta' => delta,
        'pre_ordinals' => pre_ordinals,
        'post_ordinals' => post_ordinals,
        'pre_cardinality' => pre_ordinals.length,
        'post_cardinality' => post_ordinals.length
      }
    end
  end

  def aggregate_snapshot(rows, path)
    result = {}
    rows.each do |row|
      key = [row['slot_size'], row['pcs']]
      current = result[key]
      if current && current['functions'] != row['functions']
        invalid("#{path}.rows[#{row['ordinal']}].frames", 'function stack differs for a colliding raw key')
      end
      current ||= {'counters' => zero_profile, 'ordinals' => [], 'functions' => row['functions']}
      current['counters'] = profile_add(current['counters'], row['counters'], "#{path}.raw_key")
      current['ordinals'] << row['ordinal']
      result[key] = current
    end
    result
  end

  def symbolization_for(pre, post, index)
    if pre && post && pre['functions'] != post['functions']
      invalid("$derived.raw_keys[#{index}].functions", 'function stack differs between snapshots for the same raw key')
    end
    (pre || post)['functions']
  end

  def build_normalized_sites(raw_keys)
    buckets = {}
    raw_keys.each do |raw|
      key = [raw['slot_size'], raw['functions']]
      bucket = buckets[key] ||= {
        'pre' => zero_profile,
        'post' => zero_profile,
        'delta' => zero_profile,
        'indices' => [],
        'group' => raw['group']
      }
      bucket['pre'] = profile_add(bucket['pre'], raw['pre'], '$derived.normalized_sites.pre')
      bucket['post'] = profile_add(bucket['post'], raw['post'], '$derived.normalized_sites.post')
      bucket['delta'] = profile_add(bucket['delta'], raw['delta'], '$derived.normalized_sites.delta')
      bucket['indices'] << raw['index']
    end

    buckets.keys.sort.each_with_index.map do |key, index|
      bucket = buckets[key]
      indices = bucket['indices'].sort
      {
        'index' => index,
        'slot_size' => key[0],
        'functions' => key[1].dup,
        'group' => bucket['group'],
        'pre' => bucket['pre'],
        'post' => bucket['post'],
        'delta' => bucket['delta'],
        'raw_key_indices' => indices,
        'raw_key_cardinality' => indices.length
      }
    end
  end

  def build_groups(raw_keys, normalized_sites)
    %w[caller worker_temporally_associated other].each_with_object({}) do |group, result|
      sites = normalized_sites.select { |site| site['group'] == group }
      group_raw = raw_keys.select { |raw| raw['group'] == group }
      result[group] = {
        'pre' => sum_profiles(sites, 'pre', "$derived.groups.#{group}.pre"),
        'post' => sum_profiles(sites, 'post', "$derived.groups.#{group}.post"),
        'delta' => sum_profiles(sites, 'delta', "$derived.groups.#{group}.delta"),
        'raw_key_count' => group_raw.length,
        'normalized_site_count' => sites.length
      }
    end
  end

  def totals_from_groups(groups)
    %w[pre post delta].each_with_object({}) do |kind, result|
      result[kind] = %w[caller worker_temporally_associated other].each_with_index.reduce(zero_profile) do |sum, (group, index)|
        profile_add(sum, groups[group][kind], "$derived.profile_totals.#{kind}.group[#{index}]")
      end
    end
  end

  def build_windows(boundaries)
    pairs = {
      'pre_snapshot' => %w[pre_raw_before start],
      'region' => %w[start end],
      'post_gc' => %w[end post_gc],
      'post_snapshot' => %w[post_gc post_raw],
      'serialization' => %w[post_raw tail]
    }
    result = {}
    pairs.each do |name, pair|
      result[name] = counter_sub(boundaries[pair[1]], boundaries[pair[0]], "$derived.windows.#{name}")
    end
    unless result['pre_snapshot']['total_alloc'].zero? && result['pre_snapshot']['mallocs'].zero?
      invalid('$derived.windows.pre_snapshot', 'total_alloc and mallocs must not move during the pre snapshot')
    end
    result
  end

  def group_for(functions, caller, worker)
    return 'caller' if functions.include?(caller)
    return 'worker_temporally_associated' if functions.include?(worker)
    'other'
  end

  def sum_profiles(objects, kind, path)
    objects.each_with_index.reduce(zero_profile) do |sum, (object, index)|
      profile_add(sum, object[kind], "#{path}[#{index}]")
    end
  end

  def profile_add(left, right, path)
    PROFILE_FIELDS.each_with_object({}) do |field, result|
      result[field] = checked_add(left[field], right[field], "#{path}.#{field}")
    end
  end

  def profile_sub(left, right, path, require_monotone)
    PROFILE_FIELDS.each_with_object({}) do |field, result|
      if require_monotone && left[field] < right[field]
        invalid("#{path}.#{field}", 'decreasing cumulative profile counter')
      end
      result[field] = checked_sub(left[field], right[field], "#{path}.#{field}")
    end
  end

  def counter_sub(left, right, path)
    COUNTER_FIELDS.each_with_object({}) do |field, result|
      result[field] = checked_sub(left[field], right[field], "#{path}.#{field}")
    end
  end

  def zero_profile
    PROFILE_FIELDS.each_with_object({}) { |field, result| result[field] = 0 }
  end

  def checked_add(left, right, path)
    checked_s64(left + right, path)
  end

  def checked_sub(left, right, path)
    checked_s64(left - right, path)
  end

  def checked_mul(left, right, path)
    checked_s64(left * right, path)
  end

  def checked_s64(value, path)
    invalid(path, 'signed 64-bit arithmetic overflow') unless value.between?(S64_MIN, S64_MAX)
    value
  end

  def object!(value, path, keys)
    invalid(path, 'must be an object') unless value.is_a?(Hash)
    actual = value.keys
    missing = keys - actual
    extra = actual - keys
    invalid(path, "missing keys: #{missing.join(', ')}") unless missing.empty?
    invalid(path, "unknown keys: #{extra.join(', ')}") unless extra.empty?
    value
  end

  def array!(value, path)
    invalid(path, 'must be an array') unless value.is_a?(Array)
    value
  end

  def string!(value, path)
    invalid(path, 'must be a string') unless value.is_a?(String)
    value
  end

  def nonempty_string!(value, path)
    string!(value, path)
    invalid(path, 'must be nonempty') if value.empty?
    value
  end

  def integer!(value, path, minimum, maximum)
    invalid(path, 'must be an integer') unless value.is_a?(Integer)
    invalid(path, "must be at least #{minimum}") if value < minimum
    invalid(path, "must be no greater than #{maximum}") if maximum && value > maximum
    value
  end

  def literal!(value, expected, path)
    unless value.class == expected.class && value == expected
      invalid(path, "must equal #{expected.inspect}")
    end
    value
  end

  def canonical_pc!(value, path)
    string!(value, path)
    unless /\A(?:0x0|0x[1-9a-f][0-9a-f]*)\z/.match?(value)
      invalid(path, 'must be canonical lowercase unpadded hexadecimal')
    end
    number = value[2..-1].to_i(16)
    invalid(path, 'exceeds uint64') if number > U64_MAX
    number
  end

  def invalid(path, reason)
    raise Invalid, "#{path}: #{reason}"
  end
end

if $PROGRAM_NAME == __FILE__
  begin
    unless ARGV.length == 3
      raise AllocationSites::Invalid, 'usage: analyze_sites.rb CAPTURE_JSON TAIL_JSON DERIVED_JSON'
    end
    capture_path, tail_path, output_path = ARGV
    [capture_path, tail_path, output_path].each_with_index do |path, index|
      label = %w[CAPTURE_JSON TAIL_JSON DERIVED_JSON][index]
      unless path.start_with?('/')
        raise AllocationSites::Invalid, "#{label} must be an absolute path"
      end
    end

    capture = AllocationSites.parse(File.binread(capture_path))
    tail = AllocationSites.parse(File.binread(tail_path))
    derived = AllocationSites.analyze(capture, tail)
    File.open(output_path, File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
      file.write(JSON.generate(derived))
      file.write("\n")
    end
  rescue AllocationSites::Invalid, SystemCallError, IOError => e
    warn "analyze_sites: #{e.message}"
    exit 1
  end
end
